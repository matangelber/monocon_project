import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_statistics(statistics, output_path):
    # Save statistics to a CSV file
    csv_output_path = os.path.join(output_path, "offset_2d_3d_statistics.csv")
    # Create an image of the statistics
    image_output_path = os.path.join(output_path, "statistics_image.png")

    # Flatten statistics dictionary for saving
    flattened_stats = {
        "Metric": [],
        "Delta X": [],
        "Delta Y": []
    }
    for stat in ["mean", "std", "min", "max", "percentiles"]:
        if stat == "percentiles":
            for i, perc in enumerate(["25th", "50th", "75th"]):
                flattened_stats["Metric"].append(f"{perc} Percentile")
                flattened_stats["Delta X"].append(statistics["delta_x"]["percentiles"][i])
                flattened_stats["Delta Y"].append(statistics["delta_y"]["percentiles"][i])
        else:
            flattened_stats["Metric"].append(stat)
            flattened_stats["Delta X"].append(statistics["delta_x"][stat])
            flattened_stats["Delta Y"].append(statistics["delta_y"][stat])

    # Fill in missing values for percentiles
    flattened_stats["Delta X"].extend([None] * (len(flattened_stats["Metric"]) - len(flattened_stats["Delta X"])))
    flattened_stats["Delta Y"].extend([None] * (len(flattened_stats["Metric"]) - len(flattened_stats["Delta Y"])))

    # Save to CSV
    df = pd.DataFrame(flattened_stats)
    df.to_csv(csv_output_path, index=False)


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    table_data = [["Metric", "Delta X", "Delta Y"]]
    table_data.extend([[m, dx, dy] for m, dx, dy in
                       zip(flattened_stats["Metric"], flattened_stats["Delta X"], flattened_stats["Delta Y"])])

    table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    plt.tight_layout()
    plt.savefig(image_output_path, bbox_inches="tight")
    plt.close()

    csv_output_path, image_output_path


def get_statistics(input_file, output_path):
    if not(os.path.exists(output_path)):
        os.makedirs(output_path)
    with open(input_file, 'r') as f:
        data = json.load(f)

    if len(data['annotations']) == 0:
        raise AssertionError('no annotations in json file.')
    centers_2d = []
    centers_3d = []
    min_x = 10000
    min_y = 10000
    max_x = -10000
    max_y = -10000
    a = 0
    z = 0
    d = 0
    for ann in data['annotations']:
        if ann['category_name'] != 'Car':
            continue
        tx,ty,bx,by = ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]
        # print(tx,ty,bx,by)
        if tx < min_x:
            min_x = tx
        if ty < min_y:
            min_y = ty
        if tx > max_x:
            max_x = tx
        if ty > max_y:
            max_y = ty
        center_2d = np.array(ann['bbox'][:2]) + np.array(ann['bbox'][2:]) / 2
        center_3d = ann['center2d'][:2]
        image_shape = np.array([[im['width'],im['height']]  for im in data['images'] if im['id'] == ann['image_id']][0])
        # if (tx < 1) or (ty < 1):
        #     print("AAAAAAAAAAAA")
        if (tx == 0) or (ty == 0):
            d = 1
            a += 1
            print("a: ", a)
        # if (bx == image_shape[0]) or (by == image_shape[1]):
        #     print("DDDDDDDDDDDDDD")
        if (bx == image_shape[0] - 1) or (by == image_shape[1] - 1):
            d = 1
            z += 1
            print("zzz: ", z)
            # print("EEEEEEEEEEEEEEE")
        # if tx < 0 or ty < 0 or  bx > image_shape[0] or by > image_shape[1]:
        #     print("BBBBBBBBBBBBBB")
        # if (np.abs(bx - image_shape[0]) < 10) or (np.abs(by - image_shape[1]) < 10):
            # print(bx - image_shape[0], by - image_shape[1])
            # print(bx, image_shape[0], by, image_shape[1])
            # print("FFFFFFFFFFFFFFFFFFF")
        if d == 1:
            d = 0
            continue
        # print(center_3d - center_2d)
        centers_2d.append(center_2d)
        centers_3d.append(center_3d)


    centers_2d = np.array(centers_2d)
    centers_3d = np.array(centers_3d)

    deltas = centers_3d - centers_2d
    deltas_array = np.zeros((30, 90))
    for (x,y) in deltas:
        deltas_array[int(27 + y), int(47 + x)] += 1
    delta_x = deltas[:, 0]
    delta_y = deltas[:, 1]

    print("----------------------------")
    print(min_x)
    print(min_y)
    print(max_x)
    print(max_y)
    print(delta_x.min())
    print(delta_x.max())
    print(delta_y.min())
    print(delta_y.max())
    statistics = {
        "delta_x": {
            "mean": np.mean(delta_x),
            "std": np.std(delta_x),
            "min": np.min(delta_x),
            "max": np.max(delta_x),
            "percentiles": np.percentile(delta_x, [25, 50, 75])
        },
        "delta_y": {
            "mean": np.mean(delta_y),
            "std": np.std(delta_y),
            "min": np.min(delta_y),
            "max": np.max(delta_y),
            "percentiles": np.percentile(delta_y, [25, 50, 75])
        }
    }

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bins = 240
    axes[0].hist(delta_x, bins=bins, color='blue', alpha=0.7, label="Delta X")
    axes[0].set_title("Histogram of Delta X")
    axes[0].set_xlabel("Delta X")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(delta_y, bins=bins, color='green', alpha=0.7, label="Delta Y")
    axes[1].set_title("Histogram of Delta Y")
    axes[1].set_xlabel("Delta Y")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()


    # Save the plot to an image
    output_file = os.path.join(output_path, "offset_histogram_plot.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    fig, axes = plt.subplots(1, 1, figsize=(12, 12))
    axes.hist(np.sqrt(delta_y**2 + delta_x**2), bins=bins, color='green', alpha=0.7, label="Delta Y")
    axes.set_title("Histogram 3D->2D offset")
    axes.set_xlabel("Distance")
    axes.set_ylabel("Frequency")
    axes.legend()
    # Save the plot to an image
    output_file = os.path.join(output_path, "offset_distance_histogram_plot.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    import matplotlib.image

    matplotlib.image.imsave(os.path.join(output_path, "offset_image.png"), deltas_array)
    matplotlib.image.imsave(os.path.join(output_path, "offset_image_log.png"), np.log(deltas_array + 1))

    save_statistics(statistics, output_path)

if __name__ == '__main__':
    # Example usage
    input_dir = '/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/'
    output_dir = '/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/offset_statistics/'

    # for json_filename in ['kitti_infos_test_mono3d.coco.json',
    #                       'kitti_infos_train_mono3d.coco.json',
    #                       'kitti_infos_val_mono3d.coco.json',
    #                       'kitti_infos_trainval_mono3d.coco.json']:
    for json_filename in [
                          'kitti_infos_train_mono3d.coco.json'
    ]:
        input_file = os.path.join(input_dir, json_filename)
        output_file = os.path.join(output_dir, json_filename)
        get_statistics(input_file, output_dir)