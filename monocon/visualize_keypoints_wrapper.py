import sys
import os
from json_tools import filter_json
# Save the original sys.path
original_sys_path = sys.path.copy()

# Add 'dir2/tools' to the beginning of sys.path
tools_path = os.path.abspath('../mmdetection3d-0.14.0')
sys.path.insert(0, tools_path)
os.chdir(tools_path)
from tools.misc.visualize_stereo_keypoints import main as vis_keypoints
from json_tools import filter_json
# Adding arguments dynamically

def visualize_results_wrapper(config_file, output_dir="", show=False, train_data=False, filtered_file_names=[]):
    if filtered_file_names is not None:
        if train_data:
            json_filepath = 'data/kitti/kitti_infos_train_mono3d.coco.json'
        else:
            json_filepath = 'data/kitti/kitti_infos_val_mono3d.coco.json'
        new_json_filepath = 'data/tmp/filtered_images.coco.json'
        filter_json(json_filepath, new_json_filepath, file_names=filtered_file_names, max_images=None)
    else:
        new_json_filepath = None
    parameters = [
        config_file,             # Required positional argument
        '--output-dir', output_dir,
        '--max-images', '50'
    ]
    if show:
        parameters.append('--show')
    if train_data:
        parameters.append('--train-data')
    sys.argv.extend(parameters)
    vis_keypoints(new_json_filepath)

if __name__ == '__main__':
    config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo.py')
    output_dir = os.path.abspath('../outputs/visualizations/keypoints')
    show=False
    filtered_file_names = None #['training/image_2/000063.png'] #['training/image_2/000063.png']
    train_data = True
    visualize_results_wrapper(config_file, output_dir, show=show, filtered_file_names=filtered_file_names, train_data=train_data)

