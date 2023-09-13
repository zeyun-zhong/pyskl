# 32x3, 352, nostride
clip_len = 100
sample_rate = 1

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='MST',
        in_channels=3,
        embed_dim=64,
        patch_kernel=(3, 1),
        patch_stride=(2, 1),
        patch_padding=(1, 0),
        depth=10,
        num_heads=1,
        spatial_size=25,
        temporal_size=clip_len,
        drop_rate=0.,
        drop_path=0.3,
        dim_mul_layers=([3, 2.0], [8, 2.0]),
        pool_q_stride_scale=1,
        pool_t_stride_scale=1,
        pool_kvq_kernel=(),
        pool_kv_stride_adaptive=None,
        pool_kv_stride_as_q=True,
        mode="conv",
        pool_first=False,
        rel_pos_spatial_temporal=True,
        use_abs_pos=False,
        sep_pos_embed=False,
    ),
    cls_head=dict(type='TRHead', num_classes=60, in_channels=256, dropout=0.))

dataset_type = 'PoseDataset'
# ann_file = '/hkfs/work/workspace_haic/scratch/on3546-Datasets/NTURGBD/ntu60_3danno.pkl'
# ann_file = '/home/zhong/Documents/datasets/NTU_60/ntu60_3danno.pkl'
ann_file = '/pfs/work8/workspace/ffuc/scratch/on3546-datasets/NTURGBD/ntu60_3danno.pkl'
# ann_file = '/home/zey83752/datasets/NTURGBD/ntu60_3danno.pkl'
mode = 'copy'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomScale', scale=0.1),
    dict(type='RandomRot', theta=0.3),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=sample_rate,
         out_of_bound_opt='repeat_last', keep_tail_frames=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=sample_rate,
         out_of_bound_opt='repeat_last', keep_tail_frames=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=sample_rate,
         out_of_bound_opt='repeat_last', keep_tail_frames=True, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=64),
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline,
               split='xsub_train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline,
             split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline,
              split='xsub_val'))

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0,
                 nesterov=True)
# optimizer = dict(type='AdamW', lr=0.0004, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=optimizer,
#     paramwise_cfg=dict(
#         custom_keys={
#             'ln': dict(decay_mult=0.0),
#             'bias': dict(decay_mult=0.0),
#             'pos_embed': dict(decay_mult=0.),
#             'mask_token': dict(decay_mult=0.),
#             'cls_token': dict(decay_mult=0.)
#         }))

# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_by_epoch=True,
#     warmup_iters=5,
#     warmup_ratio=1.0 / 100,
#     min_lr_ratio=1e-6)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.001,
    gamma=0.1,
    step=[110, 120])

total_epochs = 140
checkpoint_config = dict(interval=1)
evaluation = dict(interval=4, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'),
                                       dict(type='WandbLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/mst/ntu60_xsub_3dkp/j'

auto_resume = False
find_unused_parameters = False
