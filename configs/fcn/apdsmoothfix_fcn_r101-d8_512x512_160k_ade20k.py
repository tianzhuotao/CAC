_base_ = './fcn_r50-d8_512x512_160k_ade20k.py'
model = dict(pretrained='/mnt/yfs/zhuotaotian/smtxlun2/mmsegmentation/checkpoints/resnet101_v1c_mmseg.pth', backbone=dict(depth=101))

evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

model = dict(
    decode_head=dict(
        type='FCNHeadAPDNovitRefineV2EntropySmoothFix')
)

# data = dict(samples_per_gpu=2)