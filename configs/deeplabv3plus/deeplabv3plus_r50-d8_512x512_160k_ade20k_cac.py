_base_ = './deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
model = dict(
    pretrained='./checkpoints/resnet50_v1c_mmseg.pth', backbone=dict(depth=50),
    decode_head=dict(type='DepthwiseSeparableASPPHeadCAC')
    )

checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

