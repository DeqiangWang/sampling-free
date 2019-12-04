import torch
from torch import nn, no_grad
from torch.distributed import all_reduce

from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator, make_anchor_generator_retinanet
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.data import make_init_data_loader
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from math import ceil, log
from tqdm import tqdm
import os
import json

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_div(tensor1, tensor2):
    tensor1 = tensor1.clone()
    tensor2 = tensor2.clone()
    all_reduce(tensor1)
    all_reduce(tensor2)
    return tensor1 / tensor2

def request_model_info(cfg, arch):
    return cfg.NETWORK + "_" + arch

def load_prior(cfg, arch, filename="init_prior.json"):
    if not os.path.exists(filename):
        print("Initialization file %s is not existed. Calculating prior from model (1 epoch)."%(filename))
        return None 
    model = request_model_info(cfg, arch)
    print('Find prior of model %s'%(model))
    with open(filename, 'r') as f:
        model_prior_dict = json.load(f) 
        if model in model_prior_dict:
            print('Find it. Use it to initialize the model.')
            return model_prior_dict[model] 
        else:
            print('Not find. Calculating prior from model (1 epoch).')
            return None 

def save_prior(cfg, prior, arch, filename="init_prior.json"):
    model = request_model_info(cfg, arch)

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            model_prior_dict = json.load(f)
            model_prior_dict[model] = prior
    else:
        print("Initialization file %s is not existed. Create it."%(filename))
        model_prior_dict = {model : prior}
        
    with open(filename, 'w') as f:
        json.dump(model_prior_dict, f)

    print("Priors have saved to %s."%(filename), prior)

class AdaptiveInitializer(object):
    def __init__(self, cfg, bias, arch="RetinaNet"):
        device = torch.device(cfg.MODEL.DEVICE)
        if arch == "RetinaNet":
            anchor_generator = make_anchor_generator_retinanet(cfg)
            fg_iou, bg_iou = cfg.MODEL.RETINANET.FG_IOU_THRESHOLD, cfg.MODEL.RETINANET.BG_IOU_THRESHOLD
            num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
            num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        else:
            assert arch == "RPN"
            anchor_generator = make_anchor_generator(cfg)
            fg_iou, bg_iou = cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.BG_IOU_THRESHOLD
            num_classes = 1
            num_anchors = anchor_generator.num_anchors_per_location()[0] 
    
        prior = load_prior(cfg, arch)

        if prior is not None:
            nn.init.constant_(bias, -log((1 - prior) / prior))
            return

        data_loader = make_init_data_loader(
            cfg, is_distributed=True, images_per_batch=cfg.SOLVER.IMS_PER_BATCH
        )

        proposal_matcher = Matcher(
            fg_iou,
            bg_iou,
            allow_low_quality_matches=True,
        )

        backbone = build_backbone(cfg).to(device)
        num_fg, num_all = 0, 0
        num_gpus = get_num_gpus()
    
        for images, targets, _ in tqdm(data_loader):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            h, w = images.tensors.shape[-2:]

            if num_all == 0:
                features = backbone(images.tensors)
                n, c = features[0].shape[:2]
                levels = len(features)
                stride = int(h / features[0].shape[2])
    
            features = [torch.zeros(n, c, int(ceil(h / (stride * 2 ** i))), int(ceil(w / (stride * 2 ** i))), device=device) for i in range(levels)]

            anchors = anchor_generator(images, features)
            anchors = [cat_boxlist(anchors_per_image).to(device) for anchors_per_image in anchors]
            
            for anchor, target in zip(anchors, targets):
                match_quality_matrix = boxlist_iou(target, anchor)
                matched_idxs = proposal_matcher(match_quality_matrix)
                num_fg_per_image, num_bg_per_image = (matched_idxs >= 0).sum(), (matched_idxs == Matcher.BELOW_LOW_THRESHOLD).sum()
                num_fg += num_fg_per_image
                num_all += num_fg_per_image + num_bg_per_image
        fg_all_ratio = reduce_div(num_fg.float(), num_all.float()).item()
        prior = fg_all_ratio / num_classes
        nn.init.constant_(bias, -log((1 - prior) / prior))
        if torch.cuda.current_device() == 0:
            save_prior(cfg, prior, arch)

class GuidedLossWeighter(object):
    def __init__(self, scale):
        self.scale = scale
     
    def __call__(self, guider, loss):
        with no_grad():
            r = guider / loss
        loss *= r * self.scale
        return loss

"""
|:-------------:|:-------------|:------------:|:-------------|:-------------:|:-------------|
|    person     | 257253       |   bicycle    | 7056         |      car      | 43533        |
|  motorcycle   | 8654         |   airplane   | 5129         |      bus      | 6061         |
|     train     | 4570         |    truck     | 9970         |     boat      | 10576        |
| traffic light | 12842        | fire hydrant | 1865         |   stop sign   | 1983         |
| parking meter | 1283         |    bench     | 9820         |     bird      | 10542        |
|      cat      | 4766         |     dog      | 5500         |     horse     | 6567         |
|     sheep     | 9223         |     cow      | 8014         |   elephant    | 5484         |
|     bear      | 1294         |    zebra     | 5269         |    giraffe    | 5128         |
|   backpack    | 8714         |   umbrella   | 11265        |    handbag    | 12342        |
|      tie      | 6448         |   suitcase   | 6112         |    frisbee    | 2681         |
|     skis      | 6623         |  snowboard   | 2681         |  sports ball  | 6299         |
|     kite      | 8802         | baseball bat | 3273         | baseball gl.. | 3747         |
|  skateboard   | 5536         |  surfboard   | 6095         | tennis racket | 4807         |
|    bottle     | 24070        |  wine glass  | 7839         |      cup      | 20574        |
|     fork      | 5474         |    knife     | 7760         |     spoon     | 6159         |
|     bowl      | 14323        |    banana    | 9195         |     apple     | 5776         |
|   sandwich    | 4356         |    orange    | 6302         |   broccoli    | 7261         |
|    carrot     | 7758         |   hot dog    | 2884         |     pizza     | 5807         |
|     donut     | 7005         |     cake     | 6296         |     chair     | 38073        |
|     couch     | 5779         | potted plant | 8631         |      bed      | 4192         |
| dining table  | 15695        |    toilet    | 4149         |      tv       | 5803         |
|    laptop     | 4960         |    mouse     | 2261         |    remote     | 5700         |
|   keyboard    | 2854         |  cell phone  | 6422         |   microwave   | 1672         |
|     oven      | 3334         |   toaster    | 225          |     sink      | 5609         |
| refrigerator  | 2634         |     book     | 24077        |     clock     | 6320         |
|     vase      | 6577         |   scissors   | 1464         |  teddy bear   | 4729         |
|  hair drier   | 198          |  toothbrush  | 1945         |               |              |
|     total     | 849949       |              |              |               |              |
"""
"""
07+12
{'__background__ ': 0, 'aeroplane': 1285, 'bicycle': 1208, 'bird': 1820, 'boat': 1397, 'bottle': 2116, 'bus': 909, 'car': 4008, 'cat': 1616, 'chair': 4338, 'cow': 1058, 'diningtable': 1057, 'dog': 2079, 'horse': 1156, 'motorbike': 1141, 'person': 15576, 'pottedplant': 1724, 'sheep': 1347, 'sofa': 1211, 'train': 984, 'tvmonitor': 1193}
"""

# do not use class information is also okay.
class ImbalancedDecider(object):
    def __init__(self, cfg, arch, num_classes):
        #coco_quantities = torch.FloatTensor([257253,7056,43533,8654,5129,6061,4570,9970,10576,12842,1865,1983,1283,9820,10542,4766,5500,6567,9223,8014,5484,1294,5269,5128,8714,11265,12342,6448,6112,2681,6623,2681,6229,8802,3273,3747,5536,6095,4807,24070,7839,20574,5474,7760,6159,14323,9195,5776,4356,6302,7261,7758,2884,5807,7005,6296,38073,5779,8631,4192,15695,4149,5803,4960,2261,5700,2854,6422,1672,3334,225,5609,2634,24077,6320,6577,1464,4729,198,1945]) 
        #voc_quantities = torch.FloatTensor([1285, 1208, 1820, 1397, 2116, 909, 4008, 1616, 4338, 1058, 1057, 2079, 1156, 1141, 15576, 1724, 1347, 1211, 984, 1193])
        #quantities = voc_quantities if "voc" in cfg.NETWORK else coco_quantities
        #class_ratios = quantities / quantities.sum() 
        prior = load_prior(cfg, arch)
        self.thresholds = prior

    def __call__(self, scores): 
        return scores > self.thresholds 
