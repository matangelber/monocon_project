_base_ = [
    '../_base_/models/monocon_dla34.py',
    '../_base_/datasets/kitti-mono3d-car-monocon_debug.py',
    '../_base_/schedules/cyclic_200e_monocon.py',
    '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(num_classes=1)
)

runner = dict(type='EpochBasedRunner', max_epochs=16)
checkpoint_config = dict(interval=8)
evaluation = dict(interval=4)
workflow = [('train', 1)]


log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(type='MlflowLoggerHook')
    ])
