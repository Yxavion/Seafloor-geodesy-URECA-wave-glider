# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:18:51 2024

@author: YXAVION

Still building, incomplete
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pyproj
import scipy
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

#%% defining functions lat lon to xyz and vincenty forward equation for WGS84
    # points would be moving in a clockwise fashion

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
            
        ### output number of points for each shape size and time taken to complete each shape
        
        obs_num = np.round(shape_length / (glider_speed*time_per_ping))
        time_taken = shape_length / (glider_speed)
        shape_arr = [shape] * len(rads)
        
        # Append the data together
        temp_df = pd.DataFrame(data = [shape_arr, rads, obs_num, time_taken]).T
        df = pd.concat([df, temp_df], axis = 0, ignore_index=True)
    
    df.rename(columns= {0:"Shape", 1:"Radius", 2:"Number of observations", 3:"Time taken"}, inplace = True)
    return df

#%% Shape create function 
# create a shape using equations, load in library needed first
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
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import scipy
    import random
    
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
    
    # Figure , Eight Curve (Leminiscate of Gerono)
    
    if str.lower(shape) == 'figure 8':
    # Figure , Eight Curve (Lemniscate of Bernoulli)
        # Lemniscate of Bernoulli

        t = np.linspace(0, 2*np.pi, num = num_pts)

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
            
            # rotate 90 degrees and concatenate
            wg_pos2 = np.matmul(wg_pos1, R_z90)
            wg_pos = np.row_stack([wg_pos1, wg_pos2])
                 
            plt.scatter(wg_pos[:, 0], wg_pos[:, 1])

    # Final output
    return wg_pos

#%% sound speed profile

sv_file = pd.read_csv('./data/sound_vel.txt', sep=r'\s+', header = None)

# convert soundspeed profile for bellhop to use
sv_file[0] = sv_file[0].abs()
# speed at depth 0 
sv0_line = scipy.interpolate.interp1d(sv_file[0], sv_file[1], fill_value = "extrapolate")
sv0 = sv0_line(0)
sv0_df = pd.DataFrame(np.array([0, sv0])).transpose()

# sound velocity profile for bellhop to use
sv = pd.concat([sv0_df, sv_file])
sv = sv.values.tolist()

# save sound velocity as a .csv for garpos
sv_df = pd.DataFrame(sv, columns=['depth', 'speed'])
sv_df.to_csv('./data/sv.csv', index=False)


#%% Parameters set for the different models

# radius = [100, 300, 500, 1000, 1500, 1800, 2000, 2500, 3000, 4000]
#shapes = ['circle', 'square', 'figure 8', 'spiral', 'random', 'clover']

test_rads = np.array([500]) # test radius
test_shapes = ['circle'] # test shapes
glider_speed = 0.1 # in m/s
time_btw_pings = 30 # time between the pings in sec
sound_model = 'harmonic mean' # bellhop or harmonic mean

# number of times to loop the shapes to get an average of the residuals
nrun_per_shape = 3 # how many times to run that shape

# Find all the number of observation points for all the different size of the shapes
shapes_obs = shape_time(test_shapes, test_rads, glider_speed, time_btw_pings)
# get a dataframe with shape, radius, number of observations, time taken in sec for survey

#%% Parameters that are set for all models
'''Data needs to be the same as the initcfg.ini file'''
# This data can be hardcoded in to speed up if wanted

# use the data that we have gotten from the gnatss fortran benchmark for transponders positions to test garpos
# have to convert to ENU based on an arbituary point for it to run in garpos

# lat, long and height(m) of all the transponders
trans_pos = pd.DataFrame(np.array([[44.832681360 ,-125.099794900, -1832.8220], # 1.1
                                       [44.817929650, -125.126649450, -1829.6450], #1.2
                                       [44.842325200, -125.134820280, -1830.7600]]), #1.3
                             columns=['lat', 'long', 'height'])
# transponder 1.1, 1.2, 1.3 has delays, 0.2s, 0.32s and 0.44s respectively

# array center position in lat long and height
arr_center_pos = trans_pos.mean()

# aribituary point set to somewhere above array center in lat long and height
arb_pt = np.array([44.8309, -125.1204, 20])

# convert to ENU using pymap3d
# defualt ellipsoid is wgs84
# first series will be E then N then U
trans_pos_ENU = pymap3d.geodetic2enu(trans_pos['lat'], trans_pos['long'], trans_pos['height'], arb_pt[0], arb_pt[1], arb_pt[2])
arr_center_ENU = pymap3d.geodetic2enu(arr_center_pos['lat'], arr_center_pos['long'], arr_center_pos['height'], arb_pt[0], arb_pt[1], arb_pt[2])

# convert trans_pos_ENU to a pandas dataframe
transponder_pos_ENU = pd.DataFrame({'E': trans_pos_ENU[0], 'N': trans_pos_ENU[1], 'U': trans_pos_ENU[2]})

#%% Find wave glider ping locations in ENU
for i, shape_obs in shapes_obs.iterrows():

    # get the antennae positions when the glider pings for a shape
    wg_ping_pos = shape_create(shape_obs['Shape'], shape_obs['Radius'], int(shape_obs['Number of observations']))

    # get the glider receiving positions
    # change this a function later
    '''# for now just put the same positions as the ping positions'''
    
    wg_receive_pos1 = wg_ping_pos
    wg_receive_pos2 = wg_ping_pos
    wg_receive_pos3 = wg_ping_pos
    
    # combine the wave glider receive positions into one long array
    wg_receive_pos = np.concatenate([wg_receive_pos1, wg_receive_pos2, wg_receive_pos3], axis=1).reshape((len(wg_receive_pos1)*3, 3))
    
    # roll, pitch, yaw (heading)for ping positions
    # have to repeat array by 3 later using np.repeat as the 3 ping positions will the same
    roll_ping = np.random.normal(0, 2, int(shape_obs['Number of observations']))
    pitch_ping = np.random.normal(0, 2, int(shape_obs['Number of observations']))
    yaw_ping = np.random.normal(0, 2, int(shape_obs['Number of observations']))
    
    # roll pitch yaw for receiving
    roll_receive = np.random.normal(0, 2, int(shape_obs['Number of observations']*3))
    pitch_receive = np.random.normal(0, 2, int(shape_obs['Number of observations']*3))
    yaw_receive = np.random.normal(0, 2, int(shape_obs['Number of observations']*3))
    
    #%% From antennae to transducer position
    ''''check https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.597532/full for the conversion of antennae position to transducer position'''
    
    # ATD offset labelled M (from config file in fortran benchmark)
    M = np.asmatrix('0.0053 ; 0 ; 0.92813')
    
    # This will be the transducer position when the glider pings
    # temporary file that will be repeated by 3 times later after the loop
    transducer_ping_pos1 = np.zeros([len(wg_ping_pos), 3])
    
    for j in range(0, len(roll_ping)): 
        
        # rotation matrix should be R = R4 * R3 * R2 * R1, matrix multiplication
        # this will be for the ping positions
        R4 = np.array(np.asmatrix('0 1 0; 1 0 0; 0 0 -1')) # final rotation
        R3 = np.array([[np.cos(np.radians(yaw_ping[j])), -np.sin(np.radians(yaw_ping[j])),0],
                       [np.sin(np.radians(yaw_ping[j])), np.cos(np.radians(yaw_ping[j])),0],
                       [0, 0, 1]]) # yaw / heading rotation
        
        R2 = np.array([[np.cos(np.radians(pitch_ping[j])),0, np.sin(np.radians(pitch_ping[j]))], 
                       [0, 1, 0],
                       [-np.sin(np.radians(pitch_ping[j])),0, np.cos(np.radians(pitch_ping[j]))]]) # pitch rotation
        
        R1 = np.array([[1, 0, 0], 
                       [0, np.cos(np.radians(roll_ping[j])), -np.sin(np.radians(roll_ping[j]))], 
                       [0, np.sin(np.radians(roll_ping[j])), np.cos(np.radians(roll_ping[j]))]])
        
        R = R4 @ R3 @ R2 @ R1 # matrix multiplication for final rotation matrix
        
        # This will be the offset of the different ENU data for the positions
        RM = R @ M 
        
        ##### This has to be repeated 3 times when making the data input file
        transducer_ping_pos1[j,] = np.transpose(np.transpose(np.asmatrix(wg_ping_pos[j,])) + RM)
        
    # repeat this 3 times to make the sending ping all the same for the 3 transponders
    transducer_ping_pos = np.repeat(transducer_ping_pos1, 3, axis=0)
    
    # This will be the transducer position for when the glider receives back the signal
    transducer_receive_pos = np.zeros([len(wg_receive_pos), 3])
        
    for j in range(0, len(roll_receive)):
        
        # rotation matrix should be R = R4 * R3 * R2 * R1, matrix multiplication
        # this will be for the ping positions
        R4 = np.array(np.asmatrix('0 1 0; 1 0 0; 0 0 -1')) # final rotation
        R3 = np.array([[np.cos(np.radians(yaw_receive[j])), -np.sin(np.radians(yaw_receive[j])),0],
                       [np.sin(np.radians(yaw_receive[j])), np.cos(np.radians(yaw_receive[j])),0],
                       [0, 0, 1]]) # yaw / heading rotation
        
        R2 = np.array([[np.cos(np.radians(pitch_receive[j])),0, np.sin(np.radians(pitch_receive[j]))], 
                       [0, 1, 0],
                       [-np.sin(np.radians(pitch_receive[j])),0, np.cos(np.radians(pitch_receive[j]))]]) # pitch rotation
        
        R1 = np.array([[1, 0, 0], 
                       [0, np.cos(np.radians(roll_receive[j])), -np.sin(np.radians(roll_receive[j]))], 
                       [0, np.sin(np.radians(roll_receive[j])), np.cos(np.radians(roll_receive[j]))]])
        
        R = R4 @ R3 @ R2 @ R1 # matrix multiplication for final rotation matrix
        
        # This will be the offset of the different ENU data for the different positions
        RM = R @ M 
        
        transducer_receive_pos[j,] = np.transpose(np.transpose(np.asmatrix(wg_receive_pos[j,])) + RM)
        
    #%% Two way travel time calculations

    # for now will just use harmonic mean of the sound speed
    twt_send = np.zeros(np.size(yaw_receive))
    twt_receive = np.zeros(np.size(yaw_receive))
    
    for i in range(0, int(len(transducer_ping_pos[:, 1])/3)): 
        
        if str.lower(sound_model) == 'harmonic mean':
            
            # distance in m for the ping positions between tranducer and transponder
            wg_transponder_dist_ping1 = np.linalg.norm(transducer_ping_pos[i*3,:] - transponder_pos_ENU.iloc[0])
            wg_transponder_dist_ping2 = np.linalg.norm(transducer_ping_pos[(i*3) + 1,:] - transponder_pos_ENU.iloc[1])
            wg_transponder_dist_ping3 = np.linalg.norm(transducer_ping_pos[(i*3) + 2,:] - transponder_pos_ENU.iloc[2])
            
            # harmonic mean speed
            sv_mean = scipy.stats.hmean(sv_file.iloc[:, 1], weights = np.repeat(4, sv_file.shape[0]))
            
            # get the travel time for the sending of the ping in sec
            # add this to twt_receive later 
            twt_send[i*3] = wg_transponder_dist_ping1 / sv_mean
            twt_send[i*3 + 1] = wg_transponder_dist_ping2 / sv_mean
            twt_send[i*3 + 2] = wg_transponder_dist_ping3 / sv_mean
            
            # distance in m for the receive positions between transducer and transponder
            wg_transponder_dist_receive1 = np.linalg.norm(transducer_receive_pos[i*3,:] - transponder_pos_ENU.iloc[0])
            wg_transponder_dist_receive2 = np.linalg.norm(transducer_receive_pos[(i*3) + 1,:] - transponder_pos_ENU.iloc[1])
            wg_transponder_dist_receive3 = np.linalg.norm(transducer_receive_pos[(i*3) + 2,:] - transponder_pos_ENU.iloc[2])
            
            # travel time for receiving back the ping in sec
            # garpos does not require delay timings in the twt
            twt_receive[i*3] = wg_transponder_dist_receive1 / sv_mean #+ 0.2
            twt_receive[i*3 + 1] = wg_transponder_dist_receive2 / sv_mean #+ 0.32
            twt_receive[i*3 + 2] = wg_transponder_dist_receive3 / sv_mean #+ 0.44
            
    twt = twt_receive + twt_send
            
    #%% Create the datafile to run in garpos
    
    # seconds from 12am of the julian date set in the config file
    start_time = 1000
    
    # sending times 
    sending_time = start_time + np.arange(0, len(wg_ping_pos) * time_btw_pings, time_btw_pings)
    ST = np.repeat(sending_time, 3) # used in the final
    
    # receiving times
    RT = ST + twt
    
    # final array will be 
    # SET	LN	MT	TT	ResiTT	TakeOff	gamma	flag	ST	ant_e0	ant_n0	ant_u0	head0	pitch0	roll0	RT	ant_e1	ant_n1	ant_u1	head1	pitch1	roll1
    
    # values that are mostly constant or does not really matter
    Set = pd.DataFrame({'SET': ['S01'] * len(twt)})
    LN = pd.DataFrame({'LN': ['L01'] * len(twt)})
    MT = pd.DataFrame({'MT': ["M11", "M12", "M13"] * len(sending_time)})
    TT = pd.DataFrame({'TT': twt})
    ResiTT = pd.DataFrame({'ResiTT': np.zeros(len(twt))})
    takeoff = pd.DataFrame({'TakeOff': np.zeros(len(twt))})
    gamma = pd.DataFrame({'gamma': np.zeros(len(twt))})
    flag = pd.DataFrame({'flag':  ['FALSE'] * len(twt)})
    
    # ST which is the pings (sending)
    ant_e0 = pd.DataFrame({'ant_e0': np.repeat(wg_ping_pos[:,0],3)})
    ant_n0 = pd.DataFrame({'ant_n0': np.repeat(wg_ping_pos[:,1],3)})
    ant_u0 = pd.DataFrame({'ant_u0': np.repeat(wg_ping_pos[:,2],3)})
    
    head0 = pd.DataFrame({'head0': np.repeat(yaw_ping ,3)})
    pitch0 = pd.DataFrame({'pitch0': np.repeat(pitch_ping ,3)})
    roll0 = pd.DataFrame({'roll0': np.repeat(roll_ping ,3)})
    
    #RT which is the receiving of pings
    ant_e1 = pd.DataFrame({'ant_e1': wg_receive_pos[:,0]})
    ant_n1 = pd.DataFrame({'ant_n1': wg_receive_pos[:,1]})
    ant_u1 = pd.DataFrame({'ant_u1': wg_receive_pos[:,2]})
    
    head1 = pd.DataFrame({'head1': yaw_receive})
    pitch1 = pd.DataFrame({'pitch1': pitch_receive})
    roll1 = pd.DataFrame({'roll1': roll_receive})
    
    #%% C'ombine all dataframes together
    
    final_data = pd.concat([Set, LN, MT, TT, ResiTT, takeoff, gamma, flag, pd.DataFrame({'ST': ST}), ant_e0, ant_n0, ant_u0, head0, pitch0, roll0, pd.DataFrame({'RT': RT}), ant_e1, ant_n1, ant_u1, head1, pitch1, roll1], axis=1)
    
    final_data.to_csv('./data/output.csv')

    
    
    
    





















    
