import argparse
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--output-dir', type=str, default="", help='directory where visualize results will be saved')
    parser.add_argument('--show', action='store_true', help='show image')
    parser.add_argument('--max-images', type=int, default=5, help='Maximum number of images to show')
    parser.add_argument('--train-data', action='store_true', help='show image')
    args = parser.parse_args()

    return args


def main(new_coco_json=None):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = False

    # Build the dataset
    if new_coco_json is not None:
        if args.train_data:
            cfg.data.train['ann_file'] = new_coco_json
        else:
            cfg.data.test['ann_file'] = new_coco_json
    if args.train_data:
        cfg.data.train.pipeline = cfg.data.test.pipeline
        dataset = build_dataset(cfg.data.train)
    else:
        dataset = build_dataset(cfg.data.test)

    # Call the modified show function
    if getattr(dataset, 'show', None) is not None:
        eval_pipeline = cfg.get('eval_pipeline', {})
        if eval_pipeline:
            dataset.show_keypoints(max_images_to_show=args.max_images, output_dir=args.output_dir, show=args.show,
                         pipeline=eval_pipeline)

        else:
            dataset.show_keypoints(max_images_to_show=args.max_images, output_dir=args.output_dir,
                         show=args.show)  # use default pipeline
    else:
        raise NotImplementedError(
            'Show is not implemented for dataset {}!'.format(
                type(dataset).__name__))


if __name__ == '__main__':
    main()
