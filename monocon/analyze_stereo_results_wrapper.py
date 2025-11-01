import sys
import os
import mmcv
from json_tools import filter_json
# Save the original sys.path
original_sys_path = sys.path.copy()

# Add 'dir2/tools' to the beginning of sys.path
tools_path = os.path.abspath('../mmdetection3d-0.14.0')
sys.path.insert(0, tools_path)
os.chdir(tools_path)
from tools.misc.visualize_bev_stereo_results import main as vis_bev_stereo_results
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
    vis_bev_stereo_results(new_json_filepath)


if __name__ == '__main__':
    config_file = os.path.abspath('./configs/monocon/stereo_configs/monocon_dla34_200e_kitti_car_stereo_debug.py')
    results = os.path.abspath('../outputs/inference_runs/results_0013/output.pkl')
    results = os.path.abspath('../outputs/inference_runs/results_0013/output.pkl')

    output_dir = os.path.abspath('../outputs/debug_bev_stereo_outputs_0013/')
    output_dir = os.path.abspath('../outputs/visualize_results/results_bev_0013/')

    results = os.path.abspath('../outputs/inference_runs/run_0013_6_8_2025/output.pkl')
    output_dir = os.path.abspath('../outputs/visualize_results/results_bev_0013_6_8_2025/')

    results = mmcv.load(results)
    gt  =
    left_images_results = results[::2]
    right_images_results = results[1::2]

    for Left_im_results, right_im_results in zip(left_images_results, right_images_results):
        left_bboxes_3d = Left_im_results['img_bbox']
        right_bboxes_3d = right_im_results['img_bbox']
        left_pred_num = left_bboxes_3d['boxes_3d'].tensor.size(0)
        right_pred_num = right_bboxes_3d['boxes_3d'].tensor.size(0)
        if left_pred_num != right_pred_num:
            print(f'{left_pred_num} != {right_pred_num}')
        print("left predictions_size: ", left_bboxes_3d['boxes_3d'].tensor.size())
        print("right predictions_size: ", right_bboxes_3d['boxes_3d'].tensor.size())

    print("Done")
    # show=False
    # filtered_file_names = None # ['training/image_2/000063.png']
    # visualize_results_wrapper(config_file, results, output_dir, show, filtered_file_names)

