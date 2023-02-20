_base_ = ['./apdsegformer_mit-b0_512x512_160k_ade20k.py']

# model settings
model = dict(
    pretrained='pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(type='SegformerHeadAPDNovitRefineV2EntropyFixLearn', in_channels=[64, 128, 320, 512]),
    )
 
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             # resize image to multiple of 32, improve SegFormer by 0.5-1.0 mIoU.
#             dict(type='ResizeToMultiple', size_divisor=32),
#             dict(type='RandomFlip'),
#             # dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
