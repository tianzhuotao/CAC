_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='/mnt/yfs/zhuotaotian/smtxlun2/mmsegmentation/checkpoints/resnet50_v1c_mmseg.pth',
    decode_head=dict(num_classes=150, type='PSPHeadAPD'), auxiliary_head=dict(num_classes=150))

