import sys
import os
# Save the original sys.path
original_sys_path = sys.path.copy()

tools_path = os.path.abspath('../mmdetection3d-0.14.0')
sys.path.insert(0, tools_path)
os.chdir(tools_path)
from tools.test import main as mm3d_test

# Adding arguments dynamically
config_file = os.path.abspath('/home/matan/Projects/MonoCon/outputs/runs/stereo_runs/stereo_training_bz_8_fixed_2d_bbox_and_flip_006/monocon_dla34_200e_kitti_car_stereo.py')
config_file = os.path.abspath('/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/configs/monocon/monocon_dla34_inference_200e_kitti_car.py')
checkpoint_path = os.path.abspath('../monocon/checkpoints/monocon4_200.pth')
checkpoint_path = os.path.abspath('/home/matan/Projects/MonoCon/outputs/runs/stereo_runs/stereo_training_bz_8_fixed_2d_bbox_and_flip_006/epoch_180.pth')
workdir = os.path.abspath('../outputs/inference_runs/test_results_006/')

# config_file = os.path.abspath('/home/matan/Projects/MonoCon/outputs/runs/stereo_runs/stereo_training_bz_8_fixed_2d_bbox_and_flip_006/monocon_dla34_200e_kitti_car_stereo.py')
# checkpoint_path = os.path.abspath('/home/matan/Projects/MonoCon/outputs/runs/stereo_runs/stereo_heatmap_consist_loss_pretrained_0013/epoch_170.pth')
# workdir = os.path.abspath('../outputs/inference_runs/test_results_0013/')
#
# config_file = os.path.abspath('/home/matan/Projects/MonoCon/outputs/runs/stereo_runs/stereo_training_bz_8_fixed_2d_bbox_and_flip_006/monocon_dla34_200e_kitti_car_stereo.py')
# checkpoint_path = os.path.abspath('/home/matan/Projects/MonoCon/outputs/runs/stereo_runs/stereo_heatmap_consist_loss_factor_05_pretrained_0018/epoch_175.pth')
# workdir = os.path.abspath('../outputs/inference_runs/test_results_0018/')
if not os.path.exists(workdir):
    os.mkdir(workdir)
# Construct the list of arguments you want to add to sys.argv
sys.argv.extend([
    config_file,  # Replace with actual config file path
    checkpoint_path,  # Replace with actual checkpoint file path
    '--out', os.path.join(workdir, 'output.pkl'),
    # '--fuse-conv-bn',
    # '--format-only',
    '--eval', 'mAP', # , 'bbox', 'segm',
    # '--show',
    # '--show-dir', workdir,
    # '--gpu-collect',
    # '--tmpdir', 'tmp/',
    # '--seed', '42',
    # '--deterministic',
    # '--cfg-options', 'model.depth=50', 'train.epochs=100',
    # '--eval-options', 'metric=mAP',
    # '--launcher', 'pytorch',
    # '--local_rank', '0'
])

mm3d_test()
