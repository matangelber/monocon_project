import sys
import os
# Save the original sys.path
original_sys_path = sys.path.copy()

# Add 'dir2/tools' to the beginning of sys.path
tools_path = os.path.abspath('../mmdetection3d-0.14.0')
sys.path.insert(0, tools_path)
os.chdir(tools_path)
from tools.train import main as mm3d_train

# Adding arguments dynamically
config_file = os.path.abspath('./configs/monocon/monocon_dla34_200e_kitti_car_stereo_debug.py')
# config_file = os.path.abspath('./configs/monocon/monocon_dla34_200e_kitti_car_debug.py')

workdir = os.path.abspath('../outputs/debug_outputs_stereo_training')
sys.argv.extend([
    config_file,             # Required positional argument
    '--work-dir', workdir,    # Example optional arguments
    # '--resume-from', '/path/to/checkpoint.pth',
    # '--no-validate',
    # '--gpus', '2',
    # '--seed', '42',
    # '--deterministic',
    # '--cfg-options', 'model.depth=50', 'train.epochs=100'
])

mm3d_train()
