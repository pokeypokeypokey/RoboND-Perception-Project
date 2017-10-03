import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

def make_hist_feature(channels, min_val, max_val, bin_n=128):
    # Compute the histogram of the HSV channels separately
    hists = [np.histogram(c[0], bins=bin_n, range=(min_val, max_val)) for c in channels]

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate([h[0] for h in hists]).astype(np.float64)

    # Normalize the result
    hist_sum = np.sum(hist_features) or 1.
    return hist_features / hist_sum

def compute_color_histograms(cloud, using_hsv=True):
    # Compute histograms for the clusters
    point_colors = np.empty((3, cloud.width))

    # Step through each point in the point cloud
    for i, point in enumerate(pc2.read_points(cloud, skip_nans=True)):
        rgb_list = float_to_rgb(point[3])
        cols = (rgb_to_hsv(rgb_list) * 255) if using_hsv else rgb_list
        point_colors[:, i] = cols
    
    return make_hist_feature(point_colors[:, :i], 0, 256, 32)

def compute_normal_histograms(normal_cloud):
    norm_vals = np.empty((3, normal_cloud.width))

    for i, norm_component in enumerate(pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True)):
        norm_vals[:, i] = norm_component

    return make_hist_feature(norm_vals, -1, 1, 12)

def compute_size(cloud):
    first_point = next(pc2.read_points(cloud, skip_nans=True))
    min_x = max_x = first_point[0]
    min_y = max_y = first_point[1]
    min_z = max_z = first_point[2]

    for point in pc2.read_points(cloud, skip_nans=True):
        min_x = min(point[0], min_x); max_x = max(point[0], max_x)
        min_y = min(point[1], min_y); max_y = max(point[1], max_y)
        min_z = min(point[2], min_z); max_z = max(point[2], max_z)

    return (max_x - min_x, max_y - min_y, max_z - min_z)

def compute_all_features(cloud, normal_cloud):
    chists = compute_color_histograms(cloud, using_hsv=True)
    nhists = compute_normal_histograms(normal_cloud)
    return np.concatenate((chists, nhists))
    bounds = compute_size(cloud)
    return np.concatenate((chists, nhists, bounds))