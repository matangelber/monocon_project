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
    parser.add_argument(
        '--pipeline-type', type=str, choices=['train', 'test'], required=True,
        help="pipline type: choose either 'train' or 'test'.")
    args = parser.parse_args()

    return args


def main(new_coco_json=None):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.pipeline_type == 'train':
        cfg.data.test.test_mode = False
        # Build the dataset
        if new_coco_json is not None:
            cfg.data.train['ann_file'] = new_coco_json
        dataset = build_dataset(cfg.data.train)
        pipeline_name = 'train_pipeline'

    else:
        cfg.data.test.test_mode = True
        # Build the dataset
        if new_coco_json is not None:
            cfg.data.test['ann_file'] = new_coco_json
        dataset = build_dataset(cfg.data.test)
        pipeline_name = 'test_pipeline'

    # Call the modified show function
    if getattr(dataset, 'show_pipline_transform', None) is not None:
        pipeline = cfg.get(pipeline_name, {})
        if pipeline:
            dataset.show_pipline_transform(max_images_to_show=args.max_images, output_dir=args.output_dir, show=args.show,
                         pipeline=pipeline)

        else:
            dataset.show_pipline_transform(max_images_to_show=args.max_images, output_dir=args.output_dir,
                         show=args.show)  # use default pipeline
    else:
        raise NotImplementedError(
            'Show is not implemented for dataset {}!'.format(
                type(dataset).__name__))


if __name__ == '__main__':
    main()
