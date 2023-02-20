_base_ = [
    './upernet_swin_base_patch4_window7_512x512_160k_ade20k_'
    'pretrain_224x224_1K.py'
]
model = dict(
    pretrained='pretrain/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12),
    decode_head=dict(type='UPerHeadCAC', in_channels=[192, 384, 768, 1536], num_classes=150),
    auxiliary_head=dict(in_channels=768, num_classes=150))


evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
