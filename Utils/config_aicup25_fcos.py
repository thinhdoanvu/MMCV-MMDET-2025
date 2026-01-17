_base_ = 'fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'  # 🧠 Backbone: ResNet-50, anchor-free

# 📦 Dataset & class info
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

classes = (
    'aortic_valve'
)

# 🧠 Model: FCOS + sửa số lớp
model = dict(
    bbox_head=dict(
        num_classes=len(classes)
    )
)

# 📦 Train/Test pipeline
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),  # optional if no GT
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 🔁 Dataloaders
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline
    )
)

# 📊 Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox'
)

# 💾 Checkpointing
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP',
        rule='greater'
    )
)

# 🔧 Optimizer (FCOS mặc định dùng SGD, bạn có thể thay AdamW nếu muốn)
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.01)

# 🏃 Train config
train_cfg = dict(max_epochs=100, val_interval=1)