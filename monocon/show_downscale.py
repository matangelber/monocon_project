import matplotlib.pyplot as plt
import mmcv
import numpy as np
import cv2
import os

img_name = "000010/000010_7_PadStereo.png"
img_path = os.path.join("/home/matan/Projects/MonoCon/outputs/visualizations/pipeline_visualization", img_name)
output_path = os.path.join("/home/matan/Projects/MonoCon/outputs/visualizations/blocks_4_on_4_examples/", img_name)
img = mmcv.imread(img_path)
for y in range(0, img.shape[0], 4):
    for x in range(0, img.shape[1], 4):
        # Extract the 4x4 block
        block = img[y:y+4, x:x+4]

        # Calculate the average color of the block
        avg_color = np.mean(block, axis=(0, 1))
        # Draw a rectangle around the block
        cv2.rectangle(img, (x, y), (x+4, y+4), color=(0,0,0))

mmcv.imwrite(img, output_path)