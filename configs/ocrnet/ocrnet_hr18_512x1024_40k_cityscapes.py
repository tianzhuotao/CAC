_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    pretrained='pretrain/hrnetv2_w18-00eb2006.pth',)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
