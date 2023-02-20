_base_ = './pspnet_r50-d8_512x512_160k_ade20k.py'
model = dict(pretrained='/mnt/yfs/zhuotaotian/smtxlun2/mmsegmentation/checkpoints/resnet101_v1c_mmseg.pth', backbone=dict(depth=101),
decode_head=dict(type='PSPHeadAPD'))
