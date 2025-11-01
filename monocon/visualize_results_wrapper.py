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

def visualize_results_wrapper(config_file, results, output_dir="", show=False, filtered_file_names=[]):
    if filtered_file_names is not None:
        json_filepath = 'data/kitti/kitti_infos_val_mono3d.coco.json'
        new_json_filepath = 'data/tmp/kitti_infos_val_mono3d.coco.json'
        filter_json(json_filepath, new_json_filepath, file_names=filtered_file_names, max_images=None)
    else:
        new_json_filepath = None
    parameters = [
        config_file,             # Required positional argument
        '--result', results,    # Example optional arguments
        '--output-dir', output_dir,
    ]
    if show:
        parameters.append('--show')
    sys.argv.extend(parameters)
    vis_results(new_json_filepath)

if __name__ == '__main__':
    # config_file = os.path.abspath('./configs/monocon/monocon_dla34_200e_kitti_car_debug.py')
    # results = os.path.abspath('../outputs/inference_runs/pretrained_results/output.pkl')
    # output_dir = os.path.abspath('../outputs/debug_outputs/')
    # config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo_with_stereo_head_pretrained.py')
    # results = os.path.abspath('../outputs/inference_runs/run_0013_6_8_2025/output.pkl')
    # output_dir = os.path.abspath('../outputs/visualize_results/run_0013_6_8_2025/')

    config_file = "/home/matan/Projects/DA3D/configs/test_configs/TableVI_line2.py"
    results = "/home/matan/Projects/DA3D/work_dirs/validation_results/results.pkl"
    output_dir = "/home/matan/Projects/DA3D/work_dirs/validation_results/visualize_results"
    show=False
    filtered_file_names = None #['training/image_2/000063.png']
    visualize_results_wrapper(config_file, results, output_dir, show, filtered_file_names)

