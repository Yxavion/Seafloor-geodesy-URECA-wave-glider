# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:06:08 2024

@author: YXAVION
"""

#%% bearing of a point of relative to (0,0)

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import random

stepSize= 2 * math.pi / 99 # for the parametric equation

#Generated vertices
positions = []

t = 0
while t < 2 * math.pi:
    positions.append([1000 * math.sin(t), 1000 * math.cos(t), 0])
    t += stepSize
    
wg_pos = np.array(positions)

plt.scatter(wg_pos[:, 0], wg_pos[:, 1])


#%% bearing calculation

tan_inputs = wg_pos[:, 1]/ wg_pos[:, 0]
bearing = np.zeros(len(wg_pos))

for i, tan_input in enumerate(tan_inputs):
    if wg_pos[i, 0] > 0 and wg_pos[i, 1] > 0: # first quadrant
        bearing[i] = 90 - (math.atan(tan_input)*180/math.pi)
        
    elif wg_pos[i, 0] < 0 and wg_pos[i, 1] > 0: # second quadrant
        bearing[i] = 270 - (math.atan(tan_input)*180/math.pi)
        
    elif wg_pos[i, 0] < 0 and wg_pos[i, 1] < 0: # third quadrant
        bearing[i] = 270 - (math.atan(tan_input)*180/math.pi)
        
    elif wg_pos[i, 0] > 0 and wg_pos[i, 1] < 0: # fourth quadrant
        bearing[i] = 90 - (math.atan(tan_input)*180/math.pi)
    
    elif wg_pos[i, 0] > 0 and wg_pos[i, 1] == 0: # falls on + x axis
        bearing[i] = 90
        
    elif wg_pos[i, 0] == 0 and wg_pos[i, 1] > 0: # falls on + y axis
        bearing[i] = 0
    
    elif wg_pos[i, 0] < 0 and wg_pos[i, 1] == 0: # falls on - x axis
        bearing[i] = 270
    
    elif wg_pos[i, 0] == 0 and wg_pos[i, 1] < 0: # falls on - y axis
        bearing[i] = 180
    else:
        print('Check input')
    






