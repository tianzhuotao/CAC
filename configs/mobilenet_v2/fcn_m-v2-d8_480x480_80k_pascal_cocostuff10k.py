_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/coco-stuff10k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='./pretrain/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6)),
    decode_head=dict(
        in_channels=320,
        num_classes=171),
    auxiliary_head=dict(in_channels=96, num_classes=171))
