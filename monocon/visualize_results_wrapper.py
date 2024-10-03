import sys
import os
from json_tools import filter_json
# Save the original sys.path
original_sys_path = sys.path.copy()

# Add 'dir2/tools' to the beginning of sys.path
tools_path = os.path.abspath('../mmdetection3d-0.14.0')
sys.path.insert(0, tools_path)
os.chdir(tools_path)
from tools.misc.visualize_results import main as vis_results
from json_tools import filter_json
# Adding arguments dynamically

def visualize_results_wrapper(config_file, results, show_dir, filtered_file_names):
    if filtered_file_names is not None:
        json_filepath = 'data/kitti/kitti_infos_val_mono3d.coco.json'
        new_json_filepath = 'data/tmp/kitti_infos_val_mono3d.coco.json'
        filter_json(json_filepath, new_json_filepath, file_names=filtered_file_names, max_images=None)
    else:
        new_json_filepath = None
    sys.argv.extend([
        config_file,             # Required positional argument
        '--result', results,    # Example optional arguments
        '--show-dir', show_dir,
        # '--no-validate',
        # '--gpus', '2',
        # '--seed', '42',
        # '--deterministic',
        # '--cfg-options', 'model.depth=50', 'train.epochs=100'
    ])
    vis_results(new_json_filepath)


if __name__ == '__main__':
    config_file = os.path.abspath('./configs/monocon/monocon_dla34_200e_kitti_car.py')
    results = os.path.abspath('../outputs/pretrained_results/output.pkl')
    show_dir = os.path.abspath('../outputs/pretrained_results/')
    filtered_file_names = ['training/image_2/000063.png']
    visualize_results_wrapper(config_file, results, show_dir, filtered_file_names)

