_base_ = ['./apdsegformer_mit-b0_512x512_160k_ade20k.py']

# model settings
model = dict(
    pretrained='pretrain/mit_b2.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3]),
    decode_head=dict(type='SegformerHeadAPDNovitRefine', in_channels=[64, 128, 320, 512]),
    )
