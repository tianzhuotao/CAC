_base_ = './deeplabv3plus_r50-d8_512x512_80k_ade20k.py'
model = dict(pretrained='/mnt/yfs/zhuotaotian/smtxlun2/mmsegmentation/checkpoints/resnet101_v1c_mmseg.pth', backbone=dict(depth=101))
