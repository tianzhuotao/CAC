_base_ = './deeplabv3plus_r50-d8_512x512_4x4_40k_coco-stuff10k.py'
model = dict(
    pretrained='/mnt/yfs/zhuotaotian/smtxlun2/mmsegmentation/checkpoints/resnet101_v1c_mmseg.pth', backbone=dict(depth=101),
    decode_head=dict(type='DepthwiseSeparableASPPHead')
    )

checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
