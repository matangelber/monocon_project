import json
import os


def filter_json(input_file, output_file, max_images=64):
  """Filters a JSON file to keep only the specified number of images and their corresponding annotations.

  Args:
    input_file: The path to the input JSON file.
    output_file: The path to the output JSON file.
    max_images: The maximum number of images to keep.
  """

  with open(input_file, 'r') as f:
    data = json.load(f)

  # Filter the images
  filtered_images = data['images'][:max_images]
  filtered_data = {'images': filtered_images, 'categories': data['categories']}

  # Filter the annotations
  if len(data['annotations']) > 0:
    filtered_annotations = []
    for annotation in data['annotations']:
      if annotation['image_id'] in [image['id'] for image in filtered_images]:
        filtered_annotations.append(annotation)
    filtered_data['annotations'] = filtered_annotations
  else:
    filtered_data['annotations'] = []

  # Save the filtered data to a new JSON file
  with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=4)

# Example usage
input_dir = '/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/'
output_dir = '/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/debug_annotations/'
max_images = 64
for json_filename in ['kitti_infos_test_mono3d.coco.json',
                      'kitti_infos_train_mono3d.coco.json',
                      'kitti_infos_val_mono3d.coco.json',
                      'kitti_infos_trainval_mono3d.coco.json']:
    input_file = os.path.join(input_dir, json_filename)
    output_file = os.path.join(output_dir, json_filename)
    filter_json(input_file, output_file, max_images)