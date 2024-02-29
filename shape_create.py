# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:10:12 2024

@author: YXAVION
"""

#%% create a shape using equations, load in library needed first
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
    
    
    #%% circle 
    
    if shape == 'Circle':
        stepSize= 2 * math.pi / (num_pts-1) # for the parametric equation
        
        #Generated vertices
        positions = []
        
        t = 0
        while t < 2 * math.pi:
            positions.append([rad * math.sin(t), rad * math.cos(t), 0])
            t += stepSize
            
        wg_pos = np.array(positions)
        
        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    
    #%% Square
    if shape == 'Square':
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
    
    #%% Archemedean spiral
    
    if shape == 'Spiral':
        # default will be 5 rotations, ie 10pi
        
        # eqn is of radius r = a * theta
        # spacing between lines is a * pi
        
        # parametric equation
        
        # radius of query points
        rad_pts = np.linspace(0, rad, num = num_pts)
        
        # get the theta values of the query points
        a = rad / (num_rot*2*math.pi)
        theta_pts = rad_pts / a
        
        # x will follow rcos(theta) and y will follow rsin(theta)
        x_spiral = rad_pts * np.cos(theta_pts)
        y_spiral = rad_pts * np.sin(theta_pts)
        
        wg_pos = np.concatenate([x_spiral, y_spiral, np.zeros(np.size(x_spiral))])
        wg_pos = np.reshape(wg_pos, [num_pts, 3], order='F')
        
        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    #%% Figure , Eight Curve (Leminiscate of Gerono)
    
    if shape == 'Figure 8':
        # equation is x4 = a2(x2 – y2)
        # parametric eqn x = a sin(t), y = a sin(t) cos(t), a is the radius
        
        stepSize= 2 * math.pi / (num_pts-1) # for the parametric equation
        
        #Generated vertices
        positions = []
        
        t = 0
        while t < 2 * math.pi:
            positions.append([rad * math.sin(t), rad * math.cos(t) * math.sin(t), 0])
            t += stepSize
            
        wg_pos = np.array(positions)
        
        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    #%% Random points in a circle
    
    if shape == 'Random':
        
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
    
    #%% Final output
    
    return wg_pos
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





















