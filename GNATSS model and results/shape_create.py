# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:10:12 2024

@author: YXAVION
"""

#%% Shape create function
# create a shape using equations, load in library needed first
def shape_create(shape, rad, num_pts, num_rot = 5):
    # shape is the shape that you want the wave glider to move
        # circle, radius of rad
        # square, diagonals of rad, num_pts must be divisible by 4
        # spiral, spiral radius of rad. default 5 rotations (change using num_rot)
        # figure 8, length of figure 8 is 2*rad
        # random, random points in a circle of radius of rad
    # rad is the radius of the shape that you want to make, usually set to the radius of the transponder array
    # num_pts is the number of observation points that you want in the shape
    
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import scipy
    import random
    
    # circle 
    
    if str.lower(shape) == 'circle':
      
        
        t = np.linspace(0, 2*np.pi, num = num_pts)
            
        wg_pos = np.transpose(np.array([rad * np.sin(t), rad * np.cos(t), np.repeat(-4, num_pts)]))        
        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    
    # Square
    if str.lower(shape) == 'square':
        
        # rad is the diagonal of square, center to the corner of the square
        side_half = rad * math.sin(math.pi/4) # half of square side length
        
        query_pts_per_side = num_pts/4
        
        
        # start from the top left corner
        # if center is 0,0,0, the top left would be -side/2, side/2
        # this is going clockwise from the top left
        corner_pts = np.array([[-side_half, side_half], [side_half, side_half], 
                           [side_half, -side_half], [-side_half, -side_half]])
        
        # query points for the interpolation
        # top x coords and left y coords share the same values
        # bot x coords and right y coords share the same values
        topx_lefty = np.arange(corner_pts[0,0], corner_pts[1,0],
                               2*side_half/query_pts_per_side)
        botx_righty = np.arange(corner_pts[1,1], corner_pts[2,1],
                               -2*side_half/query_pts_per_side)
        
        # y points of the top and bot of the squarex
        topy_rightx = np.interp(topx_lefty ,corner_pts[0:2, 0], corner_pts[0:2, 1])
        boty_leftx = np.interp(botx_righty ,corner_pts[2:3, 0], corner_pts[2:3, 1])
        
        # join all the points into one array
        
        top_pts = np.column_stack([topx_lefty, topy_rightx,
                                   np.zeros(np.size(topx_lefty))])
        right_pts = np.column_stack([topy_rightx, botx_righty,
                                     np.zeros(np.size(topx_lefty))])
        bot_pts = np.column_stack([botx_righty, boty_leftx,
                                   np.zeros(np.size(topx_lefty))])
        left_pts = np.column_stack([boty_leftx, topx_lefty,
                                    np.zeros(np.size(topx_lefty))])
        
        
        # all points together
        wg_pos = np.concatenate((top_pts, right_pts, bot_pts, left_pts), axis=0)
        
        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    # Archemedean spiral
    
    if str.lower(shape) == 'spiral':
        # default will be 5 rotations, ie 10pi
        # default a = 0

        # parametric equation
        #x(θ) = (a + bθ) cos θ,
        #y(θ) = (a + bθ) sin θ
        # a = 0 as we start from 0
        b =  rad / (num_rot*2*np.pi)
  
        # find the arc length so that we can find a spacing to use for equal spacing
        # define the integral first
        def f(x):
            return np.sqrt(((b*np.cos(x) - (b*x)*np.sin(x))**2) + (b*np.sin(x) + (b*x)*np.cos(x))**2)
  
        theta_low = 0
        theta_high = num_rot*2*np.pi
        arc_len, err = scipy.integrate.quad(f, theta_low, theta_high)
  
        arc_intervals = np.linspace(0, arc_len, num = num_pts)
        thetas = np.sqrt(2 * arc_intervals / b)
  
        # x will follow rcos(theta) and y will follow rsin(theta)
        x_spiral = np.multiply(b*thetas, np.cos(thetas))
        y_spiral = np.multiply(b*thetas, np.sin(thetas))
  
        wg_pos = np.concatenate([x_spiral, y_spiral, np.zeros(np.size(x_spiral))])
        wg_pos = np.reshape(wg_pos, [num_pts, 3], order='F')
  
        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    # Figure , Eight Curve (Lemniscate of Bernoulli)
    
    if str.lower(shape) == 'figure 8':
    # Figure , Eight Curve (Lemniscate of Bernoulli)
        # Lemniscate of Bernoulli

        t = np.linspace(np.pi/2, 2.5*np.pi, num = num_pts)

        x = rad * np.cos(t) / (np.sin(t)**2 + 1)
        y = rad * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

        wg_pos = np.transpose(np.array([x, y, np.zeros(np.size(t))]))

        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    # Random points in a circle
    
    if str.lower(shape) == 'random':
        
        x_rand = np.array([])
        y_rand = np.array([])
        
        for i in range(0, num_pts): # number of points needed
            
            # from https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
            # explanation is given there
            r = rad * math.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            
            x_rand = np.append(x_rand, r * math.cos(theta))
            y_rand = np.append(y_rand, r * math.sin(theta))
            
        wg_pos = np.column_stack([x_rand, y_rand, np.zeros(np.size(x_rand))])
            
        plt.scatter(x_rand, y_rand)
    
    if str.lower(shape) == 'clover':
        # 2 figure Eight Curve (Lemniscate of Bernoulli) perpendicular to each other. 

            t = np.linspace(0, 2*np.pi, num = int(num_pts/2))

            x = rad * np.cos(t) / (np.sin(t)**2 + 1)
            y = rad * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

            wg_pos1 = np.transpose(np.array([x, y, np.zeros(np.size(t))]))
            R_z90 = np.array([[0, -1, 0],
                              [1, 0, 0],
                              [0, 0, 1]])
            
            wg_pos2 = np.matmul(wg_pos1, R_z90)
            wg_pos = np.row_stack([wg_pos1, wg_pos2])
            
            
            plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    # Final output
    
    return wg_pos









