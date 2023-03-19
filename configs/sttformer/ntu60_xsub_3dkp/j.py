model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STTFormer',
        len_parts=6,
        num_joints=25,
        num_frames=120,
        num_heads=3,
        num_channels=3,
        kernel_size=[3, 5],
        use_pes=True,
        config=[[64,  64,  16], [64,  64,  16],
                [64,  128, 32], [128, 128, 32],
                [128, 256, 64], [256, 256, 64],
                [256, 256, 64], [256, 256, 64]],
    ),
    cls_head=dict(type='TRHead', num_classes=60, in_channels=256, dropout=0.))

dataset_type = 'PoseDataset'
# ann_file = '/hkfs/work/workspace_haic/scratch/on3546-Datasets/NTURGBD/ntu60_3danno.pkl'
# ann_file = '/home/zhong/Documents/datasets/NTU_60/ntu60_3danno.pkl'
ann_file = '/pfs/work8/workspace/ffuc/scratch/on3546-datasets/NTURGBD/ntu60_3danno.pkl'
clip_len = 120
mode = 'copy'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomScale', scale=0.1),
    dict(type='RandomRot', theta=0.3),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0004, nesterov=True)
# optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2,
    warmup_ratio=1.0 / 100,
    min_lr_ratio=1e-6)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'), dict(type='WandbLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/posevit/posevit_ntu60_xsub_3dkp/j'

auto_resume = False

