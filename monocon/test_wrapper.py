import sys
import os
# Save the original sys.path
original_sys_path = sys.path.copy()

# Add 'dir2/tools' to the beginning of sys.path
tools_path = os.path.abspath('../mmdetection3d-0.14.0')
sys.path.insert(0, tools_path)
os.chdir(tools_path)
from tools.test import main as mm3d_test

# Adding arguments dynamically
config_file = os.path.abspath('../mmdetection3d-0.14.0/configs/monocon/monocon_dla34_200e_kitti_car_debug.py')
workdir = os.path.abspath('../outputs')
# Construct the list of arguments you want to add to sys.argv
sys.argv.extend([
    'config_file',  # Replace with actual config file path
    'checkpoint.pth',  # Replace with actual checkpoint file path
    # '--out', 'output.pkl',
    # '--fuse-conv-bn',
    # '--format-only',
    # '--eval', 'bbox', 'segm',
    # '--show',
    # '--show-dir', 'results/',
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
