_base_ = './fcn_hr18_512x512_80k_ade20k.py'
model = dict(
    pretrained='/mnt/yfs/zhuotaotian/smtxlun2/mmsegmentation/pretrain/hrnetv2_w48-d2186c55.pth',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        type='FCNHeadAPDNovitRefineEntropy',
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))
