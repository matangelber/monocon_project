from mmdet.models.utils.gaussian_target import get_topk_from_heatmap, get_local_maximum
import numpy as np
import PIL.Image as Image


def get_k_local_maximas(center_heatmap_pred):
    center_heatmap_pred_local_maxima = get_local_maximum(
        center_heatmap_pred.clone().detach(), kernel=3)

    *batch_dets, ys, xs = get_topk_from_heatmap(
        center_heatmap_pred_local_maxima, k=30)
    batch_scores, batch_index, batch_topk_labels = batch_dets
    return batch_index,  ys, xs

def scale_image_to_0_255(image_data):
    # Scale 0-1 values to 0-255
    if np.issubdtype(image_data.dtype, np.floating):
        if np.min(image_data) >= 0 and np.max(image_data) <= 1:
            image_data = (image_data * 255).astype(np.uint8)
        else:
            #If the float data is not between 0 and 1, perform standard normalization.
            min_val = np.min(image_data)
            max_val = np.max(image_data)
            image_data = (255 * (image_data - min_val) / (max_val - min_val)).astype(np.uint8)
    return image_data


# (np.array(Image.fromarray((255 * np.transpose(np.repeat(a_target_center_3d_consistency_heatmap[0], 3, axis=0), (1, 2, 0))).astype('uint8'), 'RGB').resize((1248, 384), Image.BICUBIC)) / 255.0).max()
def save_numpy_array_as_image(array, filename):
    """
    Saves a NumPy array of shape (c, h, w) as an RGB image.
    Handles values in the range 0-1 or 0-255.

    Args:
        array: NumPy array of shape (c, h, w), where c is the number of channels (3 for RGB).
        filename: The filename to save the image to (e.g., "image.png").
    """
    if array.shape[0] != 3:
        raise ValueError("Input array must have 3 channels (shape: (3, h, w))")

    # Transpose the array to (h, w, c) for PIL
    image_data = np.transpose(array, (1, 2, 0))

    image_data = scale_image_to_0_255(image_data)

    # Create a PIL Image object
    image = Image.fromarray(image_data, mode='RGB')

    # Save the image
    image.save(filename)

def resize_numpy_image_by_factor(image_array, factor, output_path=None):
    """
    Resizes a NumPy image array (c, h, w) by a given factor.

    Args:
        image_array: NumPy array representing the image (c, h, w).
        factor: The scaling factor (e.g., 4 for 4x enlargement).
        output_path: Optional path to save the resized image.

    Returns:
        Resized NumPy array, or None if an error occurs.
    """
    try:
        channels, height, width = image_array.shape
        image_array = scale_image_to_0_255(image_array)
        if channels == 3:  # RGB
            img = Image.fromarray(np.transpose(image_array, (1, 2, 0)), 'RGB')
        elif channels == 1:  # Grayscale
            img = Image.fromarray(image_array[0, :, :], 'L')
        else:
            raise ValueError("Array must have 1 or 3 channels (c, h, w)")

        new_width = int(img.width * factor)
        new_height = int(img.height * factor)
        resized_img = img.resize((new_width, new_height), Image.BICUBIC) # or LANCZOS, BILINEAR, BICUBIC

        if channels == 3:
            resized_array = np.transpose(np.array(resized_img), (2, 0, 1))
        else:
            resized_array = np.expand_dims(np.array(resized_img), axis=0)

        if output_path:
            resized_img.save(output_path)

        return resized_array / 255

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_rgb_with_single_channel(single_channel_array, channel='G'):
    """
    Creates an RGB array (3, w, h) from a single-channel array (1, w, h)
    where the specified channel is filled with the single-channel data
    and the other channels are zeros.

    Args:
        single_channel_array: NumPy array of shape (1, w, h).
        channel: The channel to fill ('R', 'G', or 'B').

    Returns:
        NumPy array of shape (3, w, h), or None if the input shape is incorrect
        or the channel is invalid.
    """
    if single_channel_array.shape[0] != 1:
        print("Error: Input array must have shape (1, w, h)")
        return None

    if channel not in ['R', 'G', 'B']:
        print("Error: Invalid channel. Must be 'R', 'G', or 'B'.")
        return None

    w, h = single_channel_array.shape[1], single_channel_array.shape[2]
    rgb_array = np.zeros((3, w, h), dtype=single_channel_array.dtype)

    if channel == 'R':
        rgb_array[0, :, :] = single_channel_array[0, :, :]
    elif channel == 'G':
        rgb_array[1, :, :] = single_channel_array[0, :, :]
    elif channel == 'B':
        rgb_array[2, :, :] = single_channel_array[0, :, :]

    return rgb_array