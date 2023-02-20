_base_ = './deeplabv3plus_r50-d8_769x769_80k_cityscapes.py'
model = dict(pretrained='/mnt/yfs/zhuotaotian/smtxlun2/mmsegmentation/checkpoints/resnet101_v1c_mmseg.pth', backbone=dict(depth=101))

checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
