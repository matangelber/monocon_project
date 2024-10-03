import json
import os


def filter_json(input_file, output_file, file_names=None, max_images=None):
  """Filters a JSON file to keep only the specified number of images and their corresponding annotations.

  Args:
    input_file: The path to the input JSON file.
    output_file: The path to the output JSON file.
    max_images: The maximum number of images to keep.
  """

  with open(input_file, 'r') as f:
    data = json.load(f)

  # Filter the images
  if max_images is not None:
      filtered_images = data['images'][:max_images]

  elif file_names is not None:
      filtered_images = [d for d in data['images'] if d['file_name'] in file_names]
  else:
      raise ValueError('You must provide either a file name or an image file name.')
  if len(filtered_images) == 0:
      print(f'No images found in {input_file}')
  # Filter the annotations
  filtered_annotations = []
  if len(data['annotations']) > 0:
    for annotation in data['annotations']:
      if annotation['image_id'] in [image['id'] for image in filtered_images]:
        filtered_annotations.append(annotation)
  else:
    print('Notice: no annotations of all images found')
  filtered_data = {'images': filtered_images, 'annotations': filtered_annotations, 'categories': data['categories']}
  # Save the filtered data to a new JSON file
  with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=4)


if __name__ == '__main__':
    # Example usage
    input_dir = '/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/'
    output_dir = '/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/debug_annotations/'
    max_images = None
    file_names = [
        'training\image_2000063.png'
    ]
    for json_filename in ['kitti_infos_test_mono3d.coco.json',
                          'kitti_infos_train_mono3d.coco.json',
                          'kitti_infos_val_mono3d.coco.json',
                          'kitti_infos_trainval_mono3d.coco.json']:
        input_file = os.path.join(input_dir, json_filename)
        output_file = os.path.join(output_dir, json_filename)
        filter_json(input_file, output_file, file_names=file_names, max_images=max_images)