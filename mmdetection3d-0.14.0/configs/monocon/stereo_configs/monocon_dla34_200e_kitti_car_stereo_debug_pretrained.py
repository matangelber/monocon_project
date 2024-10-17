_base_ = [
    '../../_base_/models/monocon_stereo_dla34.py',
    '../../_base_/datasets/kitti-mono3d-car-monocon_stereo_debug.py',
    '../../_base_/schedules/cyclic_200e_monocon.py',
    '../../_base_/default_runtime.py'
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
        dict(type='MlflowLoggerHook')
    ])
load_from = '/home/matan/Projects/MonoCon/outputs/runs/train_default_car_config_2/epoch_180.pth'

log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='MlflowLoggerHook', ignore_last=False, exp_name='compare_memonetum_and_lr')
    ])


# Import the custom hook in the config file
custom_imports = dict(
    imports=['mmdet3d.hooks.collate_fn_stereo_hook'],  # The filename of your custom hook file
    allow_failed_imports=False
)
custom_hooks = [
    dict(
        type='CollateFnStereoHook'
    )
]

