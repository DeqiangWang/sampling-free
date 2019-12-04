## Preparing Your Dataset

Please create a soft link of your dataset in this folder. We give instructions for COCO and PASCAL VOC.

### COCO

1. Download COCO dataset from [coco-website](http://mscoco.org/).

2. Check the structure of the dataset. It looks like:

```
coco
|_ train2017
|_ val2017
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ test2017
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ annotations
   |_ instances_train2017.json
   |_ instances_val2017.json
   |_ image_info_test-dev2017.json
```

3. Create a soft link in this folder.
```
ln -s your_coco_dataset_road coco
```

### PASCAL VOC

1. Download PASCAL VOC dataset from [darknet-mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/), with COCO-style annotation [coco-external](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip).

2. Check the structure of the dataset. It looks like:

```
voc
|_ VOC2007
|  |_ JPEGImages
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|_ VOC2012
|  |_ JPEGImages
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|_ annotations
   |_ pascal_train2007.json
   |_ pascal_test2007.json
   |_ pascal_val2007.json
   |_ pascal_train2012.json
   |_ pascal_val2012.json
```

3. Create a soft link in this folder.
```
ln -s your_voc_dataset_road voc
```
