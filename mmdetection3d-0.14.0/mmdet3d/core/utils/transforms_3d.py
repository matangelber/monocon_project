def get_delta_x_meters(P_source, P_dest):
    # Extract the translation components from the projection matrices
    t_source = P_source[0, 3] / P_source[0, 0]  # Translation component for camera 2
    t_dest = P_dest[0, 3] / P_dest[0, 0]  # Translation component for camera 3

    # Calculate the horizontal shift in pixels between camera 2 and camera 3
    delta_x = t_dest - t_source
    return delta_x

def get_delta_x_pixels(P_source, P_dest, depth):
    delta_x = (P_dest[0, 3] - P_source[0, 3])  / depth
    return delta_x
