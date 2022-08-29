# ---------------------------------------------------------------------
# Exercises from lesson 2 (object detection)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import open3d as o3d
import math
import numpy as np
import zlib

import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

# Exercise C2-4-6 : Plotting the precision-recall curve
def plot_precision_recall(): 

    # Please note: this function assumes that you have pre-computed the precions/recall value pairs from the test sequence
    #              by subsequently setting the variable configs.conf_thresh to the values 0.1 ... 0.9 and noted down the results.
    
    # Please create a 2d scatter plot of all precision/recall pairs
    # values taken from image on course
    Precision = [0.97, 0.94, 0.93, 0.92, 0.915, 0.91, 0.89, 0.87, 0.82]
    Recall = [0.738, 0.738, 0.743, 0.746, 0.746, 0.747, 0.748, 0.752, 0.754]
    plt.scatter(Recall, Precision)
    plt.show()


# Exercise C2-3-4 : Compute precision and recall
def compute_precision_recall(det_performance_all, conf_thresh=0.5):

    if len(det_performance_all)==0 :
        print("no detections for conf_thresh = " + str(conf_thresh))
        return
    
    # extract the total number of positives, true positives, false negatives and false positives
    # format of det_performance_all is [ious, center_devs, pos_negs]
    pos_negs = []
    for item in det_performance_all:
        pos_negs.append(item[2])
    pos_negs_arr = np.asarray(pos_negs)

    true_positives = sum(pos_negs_arr[:, 1])
    false_negatives = sum(pos_negs_arr[:,2])
    false_positives = sum(pos_negs_arr[:,3])
    print("TP = " + str(true_positives) + ", FP = " + str(false_positives) + ", FN = " + str(false_negatives))
    
    # compute precision
    precision = true_positives / (true_positives + false_positives)
    
    # compute recall
    recall = true_positives / (true_positives + false_negatives)

    print("precision = " + str(precision) + ", recall = " + str(recall) + ", conf_thres = " + str(conf_thresh) + "\n")


# Exercise C2-3-2 : Transform metric point coordinates to BEV space
def pcl_to_bev(lidar_pcl, configs, vis=True):

    # compute bev-map discretization by dividing x-range by the bev-image height
# how many meters in world space correspond to how many pixels in BEV space
# (lower boundary in x - upper boundary in x) / height parameter --> in configs
# dividing meters by pixels you get the discretization value which helps making the transformation from world space into BEV space
    lower_x = configs.lim_x[0]
    upper_x = configs.lim_x[1]
    height = configs.bev_height

    discretization = (upper_x - lower_x) / height

    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
# to make sure we don't modify the actual input point cloud
    lidar_pcl_cpy = np.copy(lidar_pcl)
# the actual metric values of all lidar points are in lidar PCL copy
# we want to transform it into de BEV pixel to do that
# we use the discretizations value from the step before
# make sure you get integer values --> numpyint_function
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / discretization))

    # transform all metrix y-coordinates as well but center the foward-facing x-axis in the middle of the image
# we can use the discretization value from the first step and some factor
    # taken from solution
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / discretization) + (configs.bev_width + 1) / 2)

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
# when you have neightboring values. That means that you have very similar coordinates in the BEV space. Be careful to negative values.
# To solve this problem you have to subtract the lower limit of the z-coordinate to avoid the shifting of the ground plane or to avoid this flipping of the color space
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] - configs.lim_z[0]

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
# look for points which fall into the same grid cell -> one coordinate in the bev-image
# get all points on one cell and convert them into just one.
# To do this we have to find out which point actually is the point we want to keep
# Use lexsort function (numpy) and sorts in ascending order to get the highest point on top, we need to put a minus on z axis so that the highest point is actually the powest point in therms of lexsort sorting order
# the return of lexsort function is an index vector and this index vector can be used for restoring the original point cloud
# rearrange the order of the points and the data point cloud with the vector returned

    # Docs: The last key in the sequence is used for the primary sort order, the second-to-last key for the secondary sort order, and so on.
    indices = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_height = lidar_pcl_cpy[indices]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    _, indices = np.unique(lidar_pcl_height[:, 0:2], axis=0, return_index=True)
    lidar_pcl_height = lidar_pcl_height[indices]

    # assign the height value of each unique entry in lidar_top_pcl to the height map and 
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
# map the stretch of height from lowest point on the road surface up to the highest defined in the config file
# and squeeze (z values) it into an 8-bit integer space which consists of only 256 values.
# The lowest point should correspond to 0 and the highest point to 256
    # taken from solution
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[np.int_(lidar_pcl_height[:, 0]), np.int_(lidar_pcl_height[:, 1])] = lidar_pcl_height[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    
    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
# creating the second channel of the BEV. In the first channel we have just put into the height.
# Now we want to put into the channel the intensity. Use the lexsort function to make sure the original lidar point is sorted in the way that it's first sorted by x and then by y and lastly by the intensity
# and get the unique
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]

    # only keep one point per grid cell
# This point has to be the one with the highest intensity
    _, indices = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_intensity = lidar_pcl_cpy[indices]

    # create the intensity map
# insert all the values which are on a list into the actual BEV image which is a 3 channel 2 dimensional matrix.
    # taken from solution
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_intensity[:, 0]), np.int_(lidar_pcl_intensity[:, 1])] = lidar_pcl_intensity[:, 3] / (np.amax(lidar_pcl_intensity[:, 3])-np.amin(lidar_pcl_intensity[:, 3]))

    # visualize intensity map
    if vis:
        img_intensity = intensity_map * 256
        img_intensity = img_intensity.astype(np.uint8)
        while (1):
            cv2.imshow('img_intensity', img_intensity)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

