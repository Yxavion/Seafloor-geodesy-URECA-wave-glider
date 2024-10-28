# -*- coding: utf-8 -*-
"""
Time check for the shapes that we want to test

Created on Wed Aug 28 10:46:37 2024

@author: YXAVION
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


#shapes = ['circle', 'square', 'figure 8', 'spiral', 'clover']
shapes = ['circle', 'square', 'figure 8', 'spiral', 'clover']
# radius = [100, 300, 500, 1000, 1500, 1800, 2000, 2500, 3000, 4000]
rads = np.array([100, 300, 500, 1000, 1500, 1800, 2000, 2500, 3000, 4000])
glider_speed = 0.1
time_per_ping = 30



#%% settings to set shapes up

def shape_time(shapes, rads, glider_speed, time_per_ping, num_rot = 5):
    '''
    returns a dataframe for the number of points for the different shape sizes and the time taken for each shape to complete
        
    Parameters:
    ===========    
    shape is the shape that you want the wave glider to move
        # circle, radius of rad
        # square, diagonals of rad, num_pts must be divisible by 4
        # spiral, spiral radius of rad. default 5 rotations (change using num_rot)
        # figure 8, length of figure 8 is 2*rad
        # clover, 2 figure 8s 90 degrees from each other
    rad is the radius of the shape that you want to make
    glider_speed is the average speed of the glider in m/s     
    num_rot is the number of rotations for shapes that can have periods such as spiral
    '''
    
    #%% Initialise the dataframe to store the values  
    
    df = pd.DataFrame()

    #%% Shapes
    # calculate perimeter of the shapes and check whether shapes can be completed  
    
    for i, shape in enumerate(shapes):
    
        if str.lower(shape) == 'circle':
            shape_length = 2*np.pi * rads
            
        if str.lower(shape) == 'square':
            
            side_half = rads * math.sin(math.pi/4) # half of square side length
            shape_length = side_half * 8
            
        if str.lower(shape) == 'spiral':
            # archemiedian spiral
            # default will be 5 rotations, ie 10pi
            
            num_rot = 5
            theta = num_rot*2*np.pi
            a = rads / theta
            shape_length = 0.5 * a *(theta* np.sqrt(1 + np.square(theta)) + np.arcsinh(theta))
            
        if str.lower(shape) == 'figure 8':
            # formula from https://mathworld.wolfram.com/Lemniscate.html
            shape_length = 5.2441151086 * rads
            
        if str.lower(shape) == 'clover':
            # 2 clovers together
            shape_length = 5.2441151086 * rads * 2
            
        #%% output number of points for each shape size and time taken to complete each shape
        
        obs_num = np.round(shape_length / (glider_speed*time_per_ping))
        time_taken = shape_length / (glider_speed)
        shape_arr = [shape] * len(rads)
        
        # Append the data together
        temp_df = pd.DataFrame(data = [shape_arr, rads, obs_num, time_taken]).T
        df = pd.concat([df, temp_df], axis = 0, ignore_index=True)
    
    df.rename(columns= {0:"Shape", 1:"Radius", 2:"Number of observation", 3:"Time taken"}, inplace = True)
    return df
    
    



