_base_ = './ocrnet_hr18_480x480_80k_lr1e-3_wd1e-4_pascal_context_59.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
model = dict(
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)),    
    pretrained='./pretrain/hrnetv2_w48-d2186c55.pth',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=59,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=59,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
