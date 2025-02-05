# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:57:56 2025

@author: YXAVION
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pyproj
import scipy
from scipy.optimize import fsolve
from scipy.spatial.distance import euclidean
import arlpy.uwapm as pm
import os
import shutil
import subprocess
import pymap3d

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

# Get the directory path of the current script
current_dir = os.path.dirname(__file__)
# Set the working directory to the current script's directory
os.chdir(current_dir)

#%% Defining geodetic functions, lat lon to xyz and vincenty forward equation for WGS84

def geodetic_to_cartesian(lat, lon, alt):
    # Define the coordinate systems
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # Perform the coordinate transformation
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return x, y, z

def cartesian_to_geodetic(x, y, z):
    # Define the coordinate systems
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # Perform the coordinate transformation
    lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
    return lat, lon, alt

def  vinc_pt(phi1, lembda1, alpha12, s) : 
        """ 
        Returns the lat and long of projected point and reverse azimuth 
        given a reference point and a distance and azimuth to project. 
        lats, longs and azimuths are passed in decimal degrees 
        Returns ( phi2,  lambda2,  alpha21 ) as a tuple
        Parameters:
        ===========
            f: flattening of the ellipsoid, set for WGS 84
            a: radius of the ellipsoid, meteres, set for WGS 84 
            phil: latitude of the start point, decimal degrees
            lembda1: longitude of the start point, decimal degrees
            alpha12: bearing, decimal degrees
            s: Distance to endpoint, meters
        NOTE: This code could have some license issues. It has been obtained 
        from a forum and its license is not clear. I'll reimplement with
        GPL3 as soon as possible.
        The code has been taken from
        https://isis.astrogeology.usgs.gov/IsisSupport/index.php?topic=408.0 (broken link)
        and refers to (broken link)
        http://wegener.mechanik.tu-darmstadt.de/GMT-Help/Archiv/att-8710/Geodetic_py
        
        Taken from Github by jtornero: https://gist.github.com/jtornero/9f3ddabc6a89f8292bb2
        """ 
        a = 6378137.0
        f = 1/298.257222101
        
        piD4 = math.atan( 1.0 ) 
        two_pi = piD4 * 8.0 
        phi1    = phi1    * piD4 / 45.0 
        lembda1 = lembda1 * piD4 / 45.0 
        alpha12 = alpha12 * piD4 / 45.0 
        if ( alpha12 < 0.0 ) : 
            alpha12 = alpha12 + two_pi 
        if ( alpha12 > two_pi ) : 
            alpha12 = alpha12 - two_pi
        b = a * (1.0 - f) 
        TanU1 = (1-f) * math.tan(phi1) 
        U1 = math.atan( TanU1 ) 
        sigma1 = math.atan2( TanU1, math.cos(alpha12) ) 
        Sinalpha = math.cos(U1) * math.sin(alpha12) 
        cosalpha_sq = 1.0 - Sinalpha * Sinalpha 
        u2 = cosalpha_sq * (a * a - b * b ) / (b * b) 
        A = 1.0 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * \
            (320 - 175 * u2) ) ) 
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2) ) ) 
        # Starting with the approx 
        sigma = (s / (b * A)) 
        last_sigma = 2.0 * sigma + 2.0   # something impossible 
            
        # Iterate the following 3 eqs unitl no sig change in sigma 
        # two_sigma_m , delta_sigma 
        while ( abs( (last_sigma - sigma) / sigma) > 1.0e-9 ):
            two_sigma_m = 2 * sigma1 + sigma 
            delta_sigma = B * math.sin(sigma) * ( math.cos(two_sigma_m) \
                    + (B/4) * (math.cos(sigma) * \
                    (-1 + 2 * math.pow( math.cos(two_sigma_m), 2 ) -  \
                    (B/6) * math.cos(two_sigma_m) * \
                    (-3 + 4 * math.pow(math.sin(sigma), 2 )) *  \
                    (-3 + 4 * math.pow( math.cos (two_sigma_m), 2 )))))
            last_sigma = sigma 
            sigma = (s / (b * A)) + delta_sigma 
        phi2 = math.atan2 ( (math.sin(U1) * math.cos(sigma) +\
            math.cos(U1) * math.sin(sigma) * math.cos(alpha12) ), \
            ((1-f) * math.sqrt( math.pow(Sinalpha, 2) +  \
            pow(math.sin(U1) * math.sin(sigma) - math.cos(U1) * \
            math.cos(sigma) * math.cos(alpha12), 2))))
        lembda = math.atan2( (math.sin(sigma) * math.sin(alpha12 )),\
            (math.cos(U1) * math.cos(sigma) -  \
            math.sin(U1) *  math.sin(sigma) * math.cos(alpha12))) 
        C = (f/16) * cosalpha_sq * (4 + f * (4 - 3 * cosalpha_sq )) 
        omega = lembda - (1-C) * f * Sinalpha *  \
            (sigma + C * math.sin(sigma) * (math.cos(two_sigma_m) + \
            C * math.cos(sigma) * (-1 + 2 *\
            math.pow(math.cos(two_sigma_m), 2) ))) 
        lembda2 = lembda1 + omega 
        alpha21 = math.atan2 ( Sinalpha, (-math.sin(U1) * \
            math.sin(sigma) +
            math.cos(U1) * math.cos(sigma) * math.cos(alpha12))) 
        alpha21 = alpha21 + two_pi / 2.0 
        if ( alpha21 < 0.0 ) : 
            alpha21 = alpha21 + two_pi 
        if ( alpha21 > two_pi ) : 
            alpha21 = alpha21 - two_pi 
        phi2 = phi2 * 45.0 / piD4 
        lembda2 = lembda2 * 45.0 / piD4 
        alpha21 = alpha21 * 45.0 / piD4
        return phi2, lembda2
    
# vincenty inverse formula from: https://www.johndcook.com/blog/2018/11/24/spheroid-distance/
from numpy import sin, cos, tan, arctan, arctan2

def ellipsoidal_distance(lat1, long1, lat2, long2):

    a = 6378137.0 # equatorial radius in meters 
    f = 1/298.257223563 # ellipsoid flattening 
    b = (1 - f)*a 
    tolerance = 1e-11 # to stop iteration

    phi1, phi2 = lat1, lat2
    U1 = arctan((1-f)*tan(phi1))
    U2 = arctan((1-f)*tan(phi2))
    L1, L2 = long1, long2
    L = L2 - L1

    lambda_old = L + 0

    while True:
    
        t = (cos(U2)*sin(lambda_old))**2
        t += (cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(lambda_old))**2
        sin_sigma = t**0.5
        cos_sigma = sin(U1)*sin(U2) + cos(U1)*cos(U2)*cos(lambda_old)
        sigma = arctan2(sin_sigma, cos_sigma) 
    
        sin_alpha = cos(U1)*cos(U2)*sin(lambda_old) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2
        cos_2sigma_m = cos_sigma - 2*sin(U1)*sin(U2)/cos_sq_alpha
        C = f*cos_sq_alpha*(4 + f*(4-3*cos_sq_alpha))/16
    
        t = sigma + C*sin_sigma*(cos_2sigma_m + C*cos_sigma*(-1 + 2*cos_2sigma_m**2))
        lambda_new = L + (1 - C)*f*sin_alpha*t
        if abs(lambda_new - lambda_old) <= tolerance:
            break
        else:
            lambda_old = lambda_new

    u2 = cos_sq_alpha*((a**2 - b**2)/b**2)
    A = 1 + (u2/16384)*(4096 + u2*(-768+u2*(320 - 175*u2)))
    B = (u2/1024)*(256 + u2*(-128 + u2*(74 - 47*u2)))
    t = cos_2sigma_m + 0.25*B*(cos_sigma*(-1 + 2*cos_2sigma_m**2))
    t -= (B/6)*cos_2sigma_m*(-3 + 4*sin_sigma**2)*(-3 + 4*cos_2sigma_m**2)
    delta_sigma = B * sin_sigma * t
    s = b*A*(sigma - delta_sigma)

    return s

# from: https://stackoverflow.com/questions/3932502/calculate-angle-between-two-latitude-longitude-points
def angleFromCoordinate(lat1, long1, lat2, long2):
    dLon = (long2 - long1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    brng = math.atan2(y, x)

    brng = math.degrees(brng)
    brng = (brng + 360) % 360
    brng = 360 - brng # count degrees clockwise - remove to make counter-clockwise

    return brng

#%% function for two way travel time from bellhop
def twt_calc(seafloor_depth, soundspeed, horizontal_dist, trans_depth):
    # must have bellhop installed and in the path
    # horizontal_dist is the horizontal dist from the wave glider to the transponder
    # trans_depth is the depth from the wave glider to the transponder
    # default wave glider source depth tried to be as small as possible
    # seafloor depth is positive
    env = pm.create_env2d(
        depth=seafloor_depth,
        soundspeed = soundspeed,
        tx_depth = 0.01, # default wave glider transmission depth,
        rx_depth = trans_depth,
        rx_range = horizontal_dist, 
        max_angle = 89.99, # straight down angle
        min_angle = 20
    )
    
    arrivals = pm.compute_arrivals(env)
    # first_arrivals = arrivals.loc[arrivals['surface_bounces'] == 0]
    # first_arrivals = first_arrivals.loc[first_arrivals['bottom_bounces'] == 0]
    
    arrival_time = min(arrivals['time_of_arrival'].to_numpy())
    
    return arrival_time

#%% functions to set shapes up with time

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
    
    #Initialise the dataframe to store the values    
    df = pd.DataFrame()

    ### Shapes
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
            
            theta = num_rot*2*np.pi
            a = np.divide(rads, theta)
            shape_length = 0.5 * a *(theta* np.sqrt(1 + np.square(theta)) + np.arcsinh(theta))
            
        if str.lower(shape) == 'figure 8':
            # formula from https://mathworld.wolfram.com/Lemniscate.html
            shape_length = 5.2441151086 * rads
            
        if str.lower(shape) == 'clover':
            # 2 clovers together
            shape_length = 5.2441151086 * rads * 2
            
        ### output number of points for each shape size and time taken to complete each shape
        
        obs_num = np.round(shape_length / (glider_speed*time_per_ping))
        time_taken = shape_length / (glider_speed)
        shape_arr = [shape] * np.size(rads)
        
        # Append the data together
        temp_df = pd.DataFrame(data = [shape_arr, rads, obs_num, time_taken]).T
        df = pd.concat([df, temp_df], axis = 0, ignore_index=True)
    
    df.rename(columns= {0:"Shape", 1:"Radius", 2:"Number of observations", 3:"Time taken"}, inplace = True)
    return df

#%% Shape create function 
# create a shape using equations
# the points returned are the positions where the glider pings 
# function for shapes around origin / array center

def shape_create(shape, rad, num_pts, num_rot = 5):
    '''
    Will return a numpy array of the xyz in cartesian, relative to the (0,0,0) of the all the points where the pings are sent out
    
    shape is the shape that you want the wave glider to move
        # circle, radius of rad
        # square, diagonals of rad, num_pts must be divisible by 4
        # spiral, spiral radius of rad. default 5 rotations (change using num_rot)
        # figure 8, length of figure 8 is 2*rad
        # random, random points in a circle of radius of rad
        # clover
    rad is the radius of the shape that you want to make, usually set to the radius of the transponder array
    num_pts is the number of observation points that you want in the shape
    '''
    # circle 
    
    if str.lower(shape) == 'circle':
      
        
        t = np.linspace(0, 2*np.pi, num = num_pts)
            
        wg_pos = np.transpose(np.array([rad * np.sin(t), rad * np.cos(t), np.repeat(0, num_pts)]))
        
        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    # Square
    if str.lower(shape) == 'square':
        # rad is the diagonal of square, center to the corner of the square
        side_half = rad * math.sin(math.pi/4) # half of square side length
        
        query_pts_per_side = num_pts//4
        
        
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
        theta_max = num_rot*2*np.pi
        b =  rad / theta_max
  
        # find the arc length so that we can find a spacing to use for equal spacing

        arc_len = 0.5 * b *(theta_max* np.sqrt(1 + np.square(theta_max)) + np.arcsinh(theta_max))
  
        arc_intervals = np.linspace(0, arc_len, num = num_pts)
        thetas = np.sqrt(2 * arc_intervals / b)
  
        # x will follow rcos(theta) and y will follow rsin(theta)
        x_spiral = np.multiply(b*thetas, np.cos(thetas))
        y_spiral = np.multiply(b*thetas, np.sin(thetas))
  
        wg_pos = np.concatenate([x_spiral, y_spiral, np.zeros(np.size(x_spiral))])
        wg_pos = np.reshape(wg_pos, [num_pts, 3], order='F')
  
        plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
    
    # Figure , Eight Curve (Leminiscate of Gerono)
    
    if str.lower(shape) == 'figure 8':
    # Figure , Eight Curve (Lemniscate of Bernoulli)
        # Lemniscate of Bernoulli, t is theta

        t = np.linspace(np.pi/2, 2.5*np.pi, num = num_pts)
        
        # find the arc length so that we can find a spacing to use for equal spacing
        arc_len = 5.2441151086 * rad
        arc_intervals = np.linspace(0, arc_len, num = num_pts)
        
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

            t = np.linspace(np.pi/2, 2.5*np.pi, num = num_pts)

            x = rad * np.cos(t) / (np.sin(t)**2 + 1)
            y = rad * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

            wg_pos1 = np.transpose(np.array([x, y, np.zeros(np.size(t))]))
            R_z90 = np.array([[0, -1, 0],
                              [1, 0, 0],
                              [0, 0, 1]])
            
            # rotate 90 degrees and concatenate
            wg_pos2 = np.matmul(wg_pos1, R_z90)
            wg_pos = np.row_stack([wg_pos1, wg_pos2])
                 
            plt.scatter(wg_pos[:, 0], wg_pos[:, 1])

    # Final output
    try:
        return wg_pos, thetas
    except:
        return wg_pos


#%% sound speed profile

sv_file = pd.read_csv('./data/sound_vel.txt', sep='\s+', header = None)

# convert soundspeed profile for bellhop to use
sv_file[0] = sv_file[0].abs()
# speed at depth 0 
sv0_line = scipy.interpolate.interp1d(sv_file[0], sv_file[1], fill_value = "extrapolate")
sv0 = sv0_line(0)
sv0_df = pd.DataFrame(np.array([0, sv0])).transpose()

sv = pd.concat([sv0_df, sv_file])
sv = sv.values.tolist()

#%% Parameters that are set for all models
    # use the data that we have gotten from the fortran benchmark for transponders positions
    # mainly used for plotting instead of calculation
    
trans1_depth = 1832.8220
trans2_depth = 1829.6450
trans3_depth = 1830.7600

trans_latlong = pd.DataFrame(np.array([[44.832681360,-125.099794900], # 1.1
                                       [44.817929650,-125.126649450], #1.2
                                       [44.842325200,-125.134820280]]), #1.3
                             columns=['lat', 'lon'])

center_depth = np.mean([trans1_depth, trans2_depth, trans3_depth])


# center of the array lat long
arr_center = np.array([44.8319, -125.1204])

# transponder positions in xyz
# transponder 1,2,3 has delays, 0.2s, 0.32s and 0.44s respectively
trans1_x, trans1_y, trans1_z = geodetic_to_cartesian(lat = trans_latlong['lat'][0], 
                                                     lon = trans_latlong['lon'][0],
                                                     alt = -trans1_depth)

trans2_x, trans2_y, trans2_z = geodetic_to_cartesian(lat = trans_latlong['lat'][1], 
                                                     lon = trans_latlong['lon'][1], 
                                                     alt = -trans2_depth)

trans3_x, trans3_y, trans3_z = geodetic_to_cartesian(lat = trans_latlong['lat'][2], 
                                                     lon = trans_latlong['lon'][2],
                                                     alt = -trans3_depth)

trans1 = np.array([trans1_x, trans1_y, trans1_z])
trans2 = np.array([trans2_x, trans2_y, trans2_z])
trans3 = np.array([trans3_x, trans3_y, trans3_z])

#%% Test codes for spiral

def find_theta_from_arc_length(s, b, theta_guess, tolerance=1e-6):
    """
    Finds the angle theta corresponding to a given arc length.
    
    Args:
      s: Arc length.
      b: Constant in the spiral equation (r = a + b*theta), a = 0.
      tolerance: Tolerance for numerical root finding.
      theta_guess is the initial thetas of glider position
    
    Returns:
      Angle theta.
    """
    def t_arc_func(theta):
        return (0.5 * b *(theta* np.sqrt(1 + np.square(theta)) + np.arcsinh(theta))) - s

    theta, _ = fsolve(t_arc_func, theta_guess, xtol=tolerance)
    return theta

# twt
def iterate_for_twt(wg_i, trans_pos, delay, glider_speed, sv_mean):
    if wg_i is None:
        raise ValueError("wg_i is None. Check the output of shape_time.")
    if trans_pos is None:
        raise ValueError("trans_pos is None. Ensure transponder positions are initialized.")
    if sv_mean is None or sv_mean <= 0:
        raise ValueError("sv_mean is invalid. Ensure the speed of sound is correctly set.") 
    
    owt_i = math.dist(wg_i, trans_pos)/sv_mean #initial owt, from wg to transponder
    owt_guess = owt_i ##initial owt guess is the same as owt_i
    #print("owt_i", owt_i)
    ## Some initialization for bookeeping etc
    #err_list = []; wg_i_list=[]; wg_guess_list = []
    err = 1000; i=0 #err is arbitrarily high so that the while statement will be recognized, i for keeping track of iterations
    
    ## And iterate until the error threshold is surpassed for each point for wg_i (wg ping positions)
    for wg_i, idx in enumerate(wg_i):
        while err >= 1e-6:
            ## guess waveglider position given initial one way travel time, delay, and guessed return one way travel time
            wg_guess = 
    
            if wg_guess is None:
                print("shape_time returned None for wg_guess. Exiting loop.")
                break
            ## Some bookeeping to keep track of things, some commented out unless errors occur
            #wg_i_list.append(wg_i)# for bookeeping
            #wg_guess_list.append(wg_guess)# for bookeeping
            #err_list.append(err)# for bookeeping
            
            ## Just a snippet to flag if things arent converging
            i+=1
            if i >= 1000:
                if i%1000 == 0:
                    print('Error, iterating excessively. Iterations:',i)
                    break
            ## Go through, calculate an error (distance between wg position between two subsequent iterations)
            err = math.dist(wg_i,wg_guess) #calculate difference between prior and subsequent guesses on waveglider position
            #calculate estimated return owt, given the new wg pos
            owt_guess = math.dist(wg_guess,trans_pos)/sv_mean 
            
            #update prior position for estimating convergence
            wg_i = wg_guess
            #print(err)
    twt = owt_i + owt_guess #output twt as the sum of the initial owt and the latest owt guess
    #print("owt_guess:", owt_guess)
    print('Two-way travel time estimated in',i,'iterations')
    return twt
        

def wg_pos_spiral(t, owt, wg_pos, trans_pos, speed, sv, rad, num_rot = 5):
    """
    Finds the angle theta corresponding to a given arc length.
    Args:
        t will be the set of thetas for the specific points of the ping positions
        owt will be the initial one way travel time of one transponder to the glider, can be an array
        Will only handle 1 transponder at a time
        wg_pos will be the ping positions of the different pings
        speed is the glider speed
        sv will be the harmonic mean sound velocity
        rad will the be radius currently being tested
    Gives back the new wg_pos and the twt
    """
    
    # Initial guess of twt and distances travelled
    twt_ini = owt * 2
    dist_ini = speed * twt_ini
    
    # initial test arc lengths
    theta_max = num_rot*2*np.pi
    b =  rad / theta_max
    test_s = (0.5 * b *(t* np.sqrt(1 + np.square(t)) + np.arcsinh(t))) + dist_ini

    # initial test thetas, positions and twt
    test_t = find_theta_from_arc_length(test_s, b, t)
    
    test_x = np.multiply(b * test_t, np.cos(test_t))
    test_y = np.multiply(b * test_t, np.sin(test_t))
    test_z = np.zeros(np.size(test_x))
    
    test_trans_pos = np.transpose(np.array([test_x, test_y, test_z]))
    
    return_dist = np.array([euclidean(test_trans_pos, wg_pos) for test_trans_pos, wg_pos in zip(test_trans_pos, wg_pos)])
    
    test_twt = owt + (return_dist / sv)

    twt = iterate_for_twt(wg_, trans_pos, delay, glider_speed, sv_mean)
    
    # new wave glider positions
    new_x = np.multiply(b * new_t, np.cos(new_t))
    new_y = np.multiply(b * new_t, np.sin(new_t))
  
    new_wg_pos = np.concatenate([new_x, new_y, np.zeros(np.size(new_x))])
    new_wg_pos = np.reshape(wg_pos, [len(new_x), 3], order='F')
    
    return new_wg_pos
    

#%%

speed = 1
time = 20

shapes_obs = shape_time(['spiral'], np.array([100]), speed, time)

for i, shape_obs in shapes_obs.iterrows():

    wg_data = shape_create(shape_obs['Shape'], shape_obs['Radius'], int(shape_obs['Number of observations']))
    
    if len(wg_data) == 2:
        
        wg_pos = wg_data[0]
        thetas = wg_data[1]
    else:
        wg_pos = wg_data
        
    

#%% test codes for lemniscate
# import numpy as np

# def lemniscate_arc_length(theta, a):
#   """
#   Calculates the arc length of a Lemniscate of Bernoulli.

#   Args:
#     theta: Angle in radians.
#     a: Constant determining the size of the lemniscate.

#   Returns:
#     Arc length of the lemniscate.
#   """
#   r = np.sqrt(2 * a**2 * np.cos(2 * theta))
#   dr_dtheta = -2 * a**2 * np.sin(2 * theta) / r
#   return a * np.sqrt(2) * scipy.special.ellipkinc(theta, 1 / np.sqrt(2))

# def find_theta_from_arc_length(s, a, tolerance=1e-6):
#   """
#   Finds the angle theta corresponding to a given arc length on the Lemniscate of Bernoulli.

#   Args:
#     s: Arc length.
#     a: Constant determining the size of the lemniscate.
#     tolerance: Tolerance for numerical root finding.

#   Returns:
#     Angle theta.
#   """
#   def func(theta):
#     return lemniscate_arc_length(theta, a) - s

#   theta_guess = s / (a * np.sqrt(2))  # Initial guess
#   theta, _ = fsolve(func, theta_guess, xtol=tolerance)
#   return theta

# def equidistant_thetas_on_lemniscate(num_points, a, total_arc_length):
#   """
#   Calculates the thetas required for equidistant points along a Lemniscate of Bernoulli.

#   Args:
#     num_points: Number of equidistant points.
#     a: Constant determining the size of the lemniscate.
#     total_arc_length: Total arc length of the lemniscate segment.

#   Returns:
#     A list of thetas for the equidistant points.
#   """
#   arc_length_step = total_arc_length / (num_points - 1)
#   thetas = []
#   for i in range(num_points):
#     arc_length = i * arc_length_step
#     theta = find_theta_from_arc_length(arc_length, a)
#     thetas.append(theta)
#   return thetas

# # Example usage
# num_points = 5
# a = 2  # Constant for the lemniscate
# # Calculate total arc length (approximation for a full lemniscate)
# total_arc_length = 4 * a * scipy.special.ellipkinc(np.pi / 4, 1 / np.sqrt(2)) 

# equidistant_thetas = equidistant_thetas_on_lemniscate(num_points, a, total_arc_length)
# print("Equidistant Thetas:", equidistant_thetas)  

    




