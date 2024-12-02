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
# config_file = os.path.abspath('./configs/monocon/monocon_dla34_200e_kitti_car_debug.py')
# config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo_debug.py')
# config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo_with_stereo_head.py')
config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo_with_stereo_head_pretrained.py')


workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_training_bz_8_no_consistency_007')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_heatmap_consist_loss_pretrained_0013')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_3d_pts_consist_loss_pretrained_0014')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_3d_pts_normalized_consist_loss_pretrained_0015')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_pretrained_0016')
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
# config_file = os.path.abspath('./configs/monocon/monocon_dla34_200e_kitti_car_debug.py')
# config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo_debug.py')
# config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo_with_stereo_head.py')
config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo_with_stereo_head_pretrained.py')


workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_training_bz_8_no_consistency_007')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_heatmap_consist_loss_pretrained_0013')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_3d_pts_consist_loss_pretrained_0014')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_3d_pts_normalized_consist_loss_pretrained_0015')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_pretrained_0016')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_heatmap_consist_loss_factor_5_pretrained_0017')
workdir = os.path.abspath('../outputs/runs/stereo_runs/stereo_heatmap_consist_loss_factor_05_pretrained_0018')






# workdir = os.path.abspath('../outputs/runs/debug_runs/lr_mom_comp_2_bz_8_stereo')

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




# workdir = os.path.abspath('../outputs/runs/debug_runs/lr_mom_comp_2_bz_8_stereo')

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