# ---------------------------------------------------------------------
# Exercises from lesson 1 (lidar)
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
import numpy as np
import zlib

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2


# Exercise C1-5-5 : Visualize intensity channel
def vis_intensity_channel(frame, lidar_name):

    print("Exercise C1-5-5")
    # extract range image from frame
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0:  # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
        print(ri.shape)

    max = np.amax(ri[:,:,0])
    min = np.amin(ri[:,:,0])
    ri[ri < 0] = 0.0

    # map value range to 8bit
    ri_range = ri[:, :, 0]
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)

    # focus on +/- 45° around the image center
    deg45 = int(img_range.shape[1] / 8)
    ri_center = int(img_range.shape[1] / 2)
    img_view = img_range[:, ri_center - deg45:ri_center + deg45]

    cv2.imshow('intensity image', img_view)
    cv2.waitKey(0)


# Exercise C1-5-2 : Compute pitch angle resolution
def print_pitch_resolution(frame, lidar_name):

    print("Exercise C1-5-2")
    # load range image
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0:  # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
        print(ri.shape)

    # compute vertical field-of-view from lidar calibration
    lidar_calib = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]
    min = lidar_calib.beam_inclination_min
    max = lidar_calib.beam_inclination_max
    vfov = max - min

    # compute pitch resolution and convert it to angular minutes
    pitch_res_rad = vfov / ri.shape[0]  # 0 - vertical, 1 - horizontal
    pitch_res_deg = pitch_res_rad * 180 / np.pi
    pitch_res_minutes = pitch_res_deg * 60
    print("pitch angle resolution = " + '{0:.2f}'.format(pitch_res_deg) + "°")
    print("pitch angle resolution = " + '{0:.2f}'.format(pitch_res_minutes) + "'")

    return

# Exercise C1-3-1 : print no. of vehicles
def print_no_of_vehicles(frame):

    print("Exercise C1-3-1")    

    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.laser_labels
    num_vehicles = 0
    for laser_label in frame.laser_labels:
        if laser_label.type == laser_label.TYPE_VEHICLE:
            num_vehicles += 1
            
    print("number of labeled vehicles in current frame = " + str(num_vehicles))