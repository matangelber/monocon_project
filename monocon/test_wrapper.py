import sys
import os
# Save the original sys.path
original_sys_path = sys.path.copy()

tools_path = os.path.abspath('../mmdetection3d-0.14.0')
sys.path.insert(0, tools_path)
os.chdir(tools_path)
from tools.test import main as mm3d_test

# Adding arguments dynamically
config_file = os.path.abspath('./configs/monocon/monocon_dla34_inference_200e_kitti_car_debug.py')
checkpoint_path = os.path.abspath('../monocon/checkpoints/monocon4_200.pth')
workdir = os.path.abspath('../outputs/pretrained_results/')
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
