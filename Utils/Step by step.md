#### 1. Convert YOLO format to COCO format  
Only images with BB must be used for converting  
```
https://github.com/thinhdoanvu/MMCV-MMDET-2025/blob/main/Utils/convert_YOLO_2_COCO.py
```

#### 2.Visualize image with BB from COCO format  
```
 https://github.com/thinhdoanvu/MMCV-MMDET-2025/blob/main/Utils/visualize_bouding_box_COCO.py
```

#### 3. Create tree folder as:
```bash
C:---\mmdetection3x
         |__data
         |____coco
         |______annotations
         |________instances_train2017.json
         |________instances_val2017.json
         |______images
         |________all images for training and validation stage
```

#### 4. Create config.py inside config folder
```bash
C:---\mmdetection3x
      |__configs
      |____config_aicup25_fasterRCNN.py
```
Copy contents to config.py:  

```bash
_base_ = 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'   #faster-rcnn Resnet: 50

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)
        # mask_head=dict(num_classes=102) #only for segmentation
    )
)


dataset_type = 'CocoDataset'

classes = ('aortic_valve')

data_root = 'data/coco/'


# Training pipeline
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,  # may be the best for training
    num_workers=4,  # may be the best for training
    persistent_workers=False,# Ensure clean worker state
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline  # fit 640x640
        )
    )


val_dataloader = dict(
    batch_size=8,   # may be the best for training
    num_workers=4,  # may be the best for training
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline  # fit 640x640
        )
    )

default_hooks = dict(   # save the best model
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater"
    )
)

# Set the maximum number of epochs for training
train_cfg = dict(max_epochs=100)
runner = dict(type='EpochBasedRunner', max_epochs=100)
```

#### 5. Model Complexity
```bash
python tools\analysis_tools\get_flops.py checkpoints\frcnn_ip102\vis_data\config.py
```
-------------------------------------------------------------------------------------------------------
Compute type: dataloader: load a picture from the dataset  
Input shape: (480, 640)  
Flops: 72.629G  
Params: 41.866M  
