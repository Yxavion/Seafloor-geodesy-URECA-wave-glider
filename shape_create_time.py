# -*- coding: utf-8 -*-
"""
Time check for the shapes that we want to test

Created on Wed Aug 28 10:46:37 2024

@author: YXAVION
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import random

#%% time with velocity
# need to calculate perimeter of the shape that needs to be drawn
# divide by the speed of the glider
# option to make it time dependent if have limited time??

#%% settings to set shapes up

# average speed of glider in m/s
glider_speed = 0.1

# time between the glider sending pulses in seconds
time_per_ping = 20 

# time limit for whole survey in hours
    # for eg time limit for whole survey
    # if there is not enough time for bigger shapes, will show up as NAN
    # will show the time needed for 
    # if no time limit, set to 'na'
time_lim = 6  # 'na' 

# mininum number of shapes that you want for the survey
min_shapes = 3
max_shapes = 10

#shapes = ['circle', 'square', 'figure 8', 'spiral', 'clover']
shape = 'circle'
# radius = [100, 300, 500, 1000, 1500, 1800, 2000, 2500, 3000, 4000]
rad = np.array([100, 300, 500, 1000, 1500, 1800, 2000, 2500, 3000, 4000])

#%% Shapes

if str.lower(shape) == 'circle':
    
    # calculate circumference and check whether shapes can be completed
    # if shapes are too large for the time limit 
    shape_length = 2*np.pi * rad
    
if str.lower(shape) == 'square':
    
    side_half = rad * math.sin(math.pi/4) # half of square side length
    shape_length = side_half * 8
    
if str.lower(shape) == 'spiral':
    
    num_rot = 5
    theta = num_rot*2*np.pi
    a = rad / theta
    shape_length = 0.5 * a *(theta* np.sqrt(1 + np.square(theta)) + np.arcsinh(theta))
    
if str.lower(shape) == 'figure 8':
    # fornula from https://mathworld.wolfram.com/Lemniscate.html
    shape_length = 5.2441151086 * rad
    
if str.lower(shape) == 'clover':
    shape_length = 5.2441151086 * rad * 2
    

# check speed and time limit to see how many complete shapes are possible
# distance coverable in time limit
distance = glider_speed * 3600 * time_lim
no_shapes_possible = np.floor(distance / shape_length)

# remove the radius that does not fit in the minimum number of shapes needed
possible_rads = rad[no_shapes_possible >= min_shapes]
possible_shape_lengths = shape_length[no_shapes_possible >= min_shapes]
# number of points per shape
num_pts = np.round(possible_shape_lengths/(glider_speed*time_per_ping))



