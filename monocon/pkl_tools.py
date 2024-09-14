import pickle
import os

def filter_pickle(input_file, output_file, max_images=64):
  """Filters a pickle file containing a list of image elements to keep only the specified number of images.

  Args:
    input_file: The path to the input pickle file.
    output_file: The path to the output pickle file (optional).
    max_images: The maximum number of images to keep.
  """

  with open(input_file, 'rb') as f:
    data = pickle.load(f)

  if not isinstance(data, list):
    raise ValueError("The input data must be a list.")

  filtered_data = data[:max_images]

  if output_file:
    with open(output_file, 'wb') as f:
      pickle.dump(filtered_data, f)

  return filtered_data

# Example usage
input_dir = '/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/'
output_dir = '/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/debug_annotations/'
max_images = 64
for json_filename in ['kitti_infos_test.pkl',
                      'kitti_infos_val.pkl',
                      'kitti_infos_train.pkl',
                      'kitti_infos_trainval.pkl']:
    input_file = os.path.join(input_dir, json_filename)
    output_file = os.path.join(output_dir, json_filename)
    filter_pickle(input_file, output_file, max_images)