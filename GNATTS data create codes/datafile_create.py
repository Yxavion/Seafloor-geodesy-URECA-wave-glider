# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:51:18 2024

@author: YXAVION

To run command prompt gnatss from python,
you MUST CHANGE env.txt file to your own computer env if gnatss is not in anaconda env

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

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Get the directory path of the current script
current_dir = os.path.dirname(__file__)
# Set the working directory to the current script's directory
os.chdir(current_dir)


#%% defining functions lat lon to xyz and vincenty forward equation for WGS84
#### have to create func to get points for a certain shape
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
        import math
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
from scipy import sin, cos, tan, arctan, arctan2

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
    
#%% Shape create function 
# create a shape using equations, load in library needed first
# function for shapes around origin / array center
def shape_create(shape, rad, num_pts, num_rot = 5):
    # shape is the shape that you want the wave glider to move
        # circle, radius of rad
        # square, diagonals of rad, num_pts must be divisible by 4
        # spiral, spiral radius of rad. default 5 rotations (change using num_rot)
        # figure 8, length of figure 8 is 2*rad
        # random, random points in a circle of radius of rad
        # clover
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
            
            wg_pos2 = np.matmul(wg_pos1, R_z90)
            wg_pos = np.row_stack([wg_pos1, wg_pos2])
            
            
            plt.scatter(wg_pos[:, 0], wg_pos[:, 1])
# Final output
    
    return wg_pos

#%% two way time function using bellhop

def twt_calc(seafloor_depth, soundspeed, horizontal_dist, trans_depth):
    # must have bellhop installed and in the path
    # horizontal_dist is the horizontal dist from the wave glider to the transponder
    # trans_depth is the depth from the wave glider to the transponder
    # default wave glider source depth tried to be as small as possible
    # seafloor depth is positive
    env = pm.create_env2d(
        depth=seafloor_depth,
        soundspeed = soundspeed,
        tx_depth = 0.00000000000000001, # default wave glider transmission depth,
        rx_depth = trans_depth,
        rx_range = horizontal_dist, 
        max_angle = 60, # straight down angle
        min_angle = 30
    )
    
    arrivals = pm.compute_arrivals(env)
    # first_arrivals = arrivals.loc[arrivals['surface_bounces'] == 0]
    # first_arrivals = first_arrivals.loc[first_arrivals['bottom_bounces'] == 0]
    
    arrival_time = min(arrivals['time_of_arrival'].to_numpy())
    
    return arrival_time

#%% sound speed profile

sv_file = pd.read_csv('./data/sound_vel.txt', delim_whitespace = True, header = None)

# convert soundspeed profile for bellhop to use
sv_file[0] = sv_file[0].abs()
# speed at depth 0 
sv0_line = scipy.interpolate.interp1d(sv_file[0], sv_file[1], fill_value = "extrapolate")
sv0 = sv0_line(0)
sv0_df = pd.DataFrame(np.array([0, sv0])).transpose()

sv = pd.concat([sv0_df, sv_file])
sv = sv.values.tolist()
    
#%% Parameters to change for the different models

# radius = [100, 300, 500, 1000, 1500, 1800, 2000, 2500, 3000, 4000]
#shapes = ['circle', 'square', 'figure 8', 'spiral', 'random', 'clover']
test_rads = np.array([100, 300, 500])
test_shapes = ['circle']

for test_shape in test_shapes:
    for test_rad in test_rads:
        rad = test_rad # radius of shape in meters
        num_pts = 500 # per shape
        # must be divisible by 4 if using square
        shape = test_shape # check shape create for all shapes
        sound_model = 'harmonic mean' # bellhop or harmonic mean
        nrun_per_shape = 3 # how many times to run that shape
        centered_around = 'array center' # array center or transponders
            # if centered around transponders, the number of points will be the tripled
        
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
        
        
        #%% Set up equation for the glider positions
            # will convert to a function later
            # function called shape_create
            # the loop will be for multiple iterations of the same shape to add some noise and test the shape thoroughly
            
        wg_pos = np.array([])
        
        if str.lower(centered_around) == 'array center':
            for i in range(0,nrun_per_shape):
                wg_pos = np.append(wg_pos, shape_create(shape, rad, num_pts))
                
            wg_pos = np.reshape(wg_pos, [int(len(wg_pos)/3), 3])
            
        elif str.lower(centered_around) == 'transponders':
            for i in range(0, nrun_per_shape):
                wg_pos = np.append(wg_pos, shape_create(shape, rad, num_pts))
                
            wg_pos = np.reshape(wg_pos, [int(len(wg_pos)/3), 3])
        
        #%% Points of the wave glider, starting from the first point, directly north
            # has to be changed to fit all the shapes
            # change the bearing of the vinc
            
        # time(no) will be the twt of the transponders for every ping
        # will be tripled if use circles around transponders
        obs_pts = len(wg_pos)
        
        # claculate bearing to each point of the
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
                
        
        if str.lower(centered_around) == 'array center':
              
            lat = np.zeros(obs_pts)
            long = np.zeros(obs_pts)
            x = np.zeros(obs_pts)
            y = np.zeros(obs_pts)
            z = np.zeros(obs_pts)
            
            for i in range(0, len(wg_pos)):
                    
                dist_moved = math.dist([0,0], wg_pos[i,0:2])
                if dist_moved == 0:
                    lat[i] = 44.8319
                    long[i] = -125.1204
                else:
                    lat[i], long[i] = vinc_pt(44.8319, -125.1204, bearing[i], dist_moved)
                    
                x[i], y[i], z[i] = geodetic_to_cartesian(lat[i], long[i], 0)
            
            wg_pos_xyz = np.transpose(np.array([x,y,z]))
            
        elif str.lower(centered_around) == 'transponders':
            
            obs_pts = obs_pts*3
            
            lat = np.zeros(obs_pts)
            long = np.zeros(obs_pts)
            x = np.zeros(obs_pts)
            y = np.zeros(obs_pts)
            z = np.zeros(obs_pts)
            
            for j in range(0,3):
                for i in range(0, len(wg_pos)):
                    dist_moved = math.dist([0,0], wg_pos[i,0:2])
                    lat[i +j*len(wg_pos)], long[i +j*len(wg_pos)] = vinc_pt(trans_latlong['lat'][j], trans_latlong['lon'][j], bearing[i], dist_moved)
                        
                    x[i +j*len(wg_pos)], y[i +j*len(wg_pos)], z[i +j*len(wg_pos)] = geodetic_to_cartesian(lat[i +j*len(wg_pos)], long[i +j*len(wg_pos)], 0)
                    
            wg_pos_xyz = np.transpose(np.array([x,y,z]))
            
        
        #%% calculate twtt for the transponders
        
        time1_clean = np.zeros(obs_pts)
        time2_clean = np.zeros(obs_pts)
        time3_clean = np.zeros(obs_pts)
        
        if str.lower(sound_model) == 'bellhop':
            for i, wg in enumerate(wg_pos_xyz):
                # distance of the glider to transponders
                dist1 = math.dist(wg, trans1)
                hor_dist1 = np.sqrt(dist1**2 - trans1_depth**2)
                time1_clean[i] = 2 * twt_calc(1880, sv, hor_dist1, trans1_depth)
                
                dist2 = math.dist(wg, trans2)
                hor_dist2 = np.sqrt(dist2**2 - trans2_depth**2)
                time2_clean[i] = 2 * twt_calc(1880, sv, hor_dist2, trans2_depth)
                
                dist3 = math.dist(wg, trans3)
                hor_dist3 = np.sqrt(dist3**2 - trans3_depth**2)
                time3_clean[i] = 2 * twt_calc(1880, sv, hor_dist3, trans2_depth)
                
                # print the iteration so that can track progress
                print(i)
                
        elif str.lower(sound_model) == 'harmonic mean':
            sv_mean = scipy.stats.hmean(sv_file.iloc[:, 1], weights = np.repeat(4, sv_file.shape[0]))
            time1_clean = np.zeros(obs_pts)
            time2_clean = np.zeros(obs_pts)
            time3_clean = np.zeros(obs_pts)
             
            for i, wg in enumerate(wg_pos_xyz):
                time1_clean[i] = math.dist(wg, trans1)/sv_mean*2
                time2_clean[i] = math.dist(wg, trans2)/sv_mean*2
                time3_clean[i] = math.dist(wg, trans3)/sv_mean*2
            
            
        #%% write twt into text file
        # add noise and delay to the signals
        # convert to ms for the file
        
        # delay is in seconds
        delay1 = 0.2
        delay2 = 0.32
        delay3 = 0.44
        
        # convert to ms
        time1 = (time1_clean + delay1 + np.random.normal(0, 0.00001, time1_clean.size))*10**6
        time2 = (time2_clean + delay2 + np.random.normal(0, 0.00001, time1_clean.size))*10**6
        time3 = (time3_clean + delay3 + np.random.normal(0, 0.00001, time1_clean.size))*10**6
        
        time1 = time1.round()
        time2 = time2.round()
        time3 = time3.round()
        
        # Write delay into text file with times
        # insert time from random starting point
        
        dates = pd.date_range(pd.Timestamp("26-JUN-23 00:00:07.00"),
                              freq="20S", periods = obs_pts)
        
        dates = dates.strftime("%d-%b-%y %H:%M:%S.%f")
        
        # Convert time array to dataframe
        
        twt_df = pd.DataFrame([time1, time2, time3, dates, np.zeros(len(time1))])
        twt_df = twt_df.T
        
        twt_df.columns=['t1','t2','t3','dates','dumb col']
        twt_df = twt_df[['dates','t1','t2','t3', 'dumb col']]
        twt_df[['t1', 't2', 't3']].map('${:,.0f}'.format)
        
        # twt_df.to_csv("./data/twt.txt", index=False, sep = '\t', header=False)
        
        twt_df_str = twt_df.to_string(formatters={'t1': '{:.d}'.format,
                                                  't2': '{:.d}'.format,
                                                  't3': '{:.d}'.format},
                                      header=False, index=False)
        
        #open text file
        text_file = open('./data/twt.txt', "w")
         
        #write string to file
        text_file.write(twt_df_str)
         
        #close file
        text_file.close()
        
        #%% Make the tranponder position file 
        # needs time in j2000, XYZ, and covar 9 values
        # covariance matrix value will be changed later. 
        
        start_time = 741009607.0000000
        
        np.random.seed(45)
        data_mat = np.array([start_time, x[0], y[0], z[0], 
                             np.random.normal(0.000622428, 0.000671889),
                             np.random.normal(-2.25e-07, 3.1569927543352893e-06),
                             np.random.normal(-1.96E-06, 4.312033572168257e-06),
                             np.random.normal(-2.25E-07, 3.1569927543352893e-06),
                             np.random.normal(0.000622267, 0.000671086),
                             np.random.normal(-2.79E-06, 6.130817316982711e-06),
                             np.random.normal(-1.96E-06, 4.312033572168257e-06),
                             np.random.normal(-2.79E-06, 6.130817316982711e-06),
                             np.random.normal(0.000622069, 0.000670237)])
                             
        
        for i, wg in enumerate(wg_pos_xyz):
        
            data_mat = np.append(data_mat, np.array([start_time + (time1[i]/1e6), x[i], y[i], z[i],                     np.random.normal(0.000622428, 0.000671889),
                                 np.random.normal(-2.25e-07, 3.1569927543352893e-06),
                                 np.random.normal(-1.96E-06, 4.312033572168257e-06),
                                 np.random.normal(-2.25E-07, 3.1569927543352893e-06),
                                 np.random.normal(0.000622267, 0.000671086),
                                 np.random.normal(-2.79E-06, 6.130817316982711e-06),
                                 np.random.normal(-1.96E-06, 4.312033572168257e-06),
                                 np.random.normal(-2.79E-06, 6.130817316982711e-06),
                                 np.random.normal(0.000622069, 0.000670237)]))
            
            data_mat = np.append(data_mat, np.array([start_time + (time2[i]/1e6), x[i], y[i], z[i], np.random.normal(0.000622428, 0.000671889),
                                 np.random.normal(-2.25e-07, 3.1569927543352893e-06),
                                 np.random.normal(-1.96E-06, 4.312033572168257e-06),
                                 np.random.normal(-2.25E-07, 3.1569927543352893e-06),
                                 np.random.normal(0.000622267, 0.000671086),
                                 np.random.normal(-2.79E-06, 6.130817316982711e-06),
                                 np.random.normal(-1.96E-06, 4.312033572168257e-06),
                                 np.random.normal(-2.79E-06, 6.130817316982711e-06),
                                 np.random.normal(0.000622069, 0.000670237)]))
            
            data_mat = np.append(data_mat, np.array([start_time + (time3[i]/1e6), x[i], y[i], z[i],                     np.random.normal(0.000622428, 0.000671889),
                                 np.random.normal(-2.25e-07, 3.1569927543352893e-06),
                                 np.random.normal(-1.96E-06, 4.312033572168257e-06),
                                 np.random.normal(-2.25E-07, 3.1569927543352893e-06),
                                 np.random.normal(0.000622267, 0.000671086),
                                 np.random.normal(-2.79E-06, 6.130817316982711e-06),
                                 np.random.normal(-1.96E-06, 4.312033572168257e-06),
                                 np.random.normal(-2.79E-06, 6.130817316982711e-06),
                                 np.random.normal(0.000622069, 0.000670237)]))
            
            start_time += 20
            
            if i == obs_pts - 1:
                break
            
            data_mat = np.append(data_mat, np.array([start_time, x[i+1], y[i+1], z[i+1],
                             np.random.normal(0.000622428, 0.000671889),
                             np.random.normal(-2.25e-07, 3.1569927543352893e-06),
                             np.random.normal(-1.96E-06, 4.312033572168257e-06),
                             np.random.normal(-2.25E-07, 3.1569927543352893e-06),
                             np.random.normal(0.000622267, 0.000671086),
                             np.random.normal(-2.79E-06, 6.130817316982711e-06),
                             np.random.normal(-1.96E-06, 4.312033572168257e-06),
                             np.random.normal(-2.79E-06, 6.130817316982711e-06),
                             np.random.normal(0.000622069, 0.000670237)]))
        
        
        data_mat_1 = np.reshape(data_mat, (int(len(data_mat)/13), 13))
        
        # # add uncertainties to geodetic cartesian coords
        # data_mat_1[:, 1:4] = data_mat_1[:, 1:4] + np.random.normal(
        #     0, 0.5, size=[len(data_mat_1[:, 1:4]), 3])
        
        
        wg_pos_df = pd.DataFrame(data_mat_1, columns=['Time', 'x', 'y', 'z', 'cov1', 'cov2','cov3','cov4','cov5','cov6','cov7','cov8','cov9'])
        wg_df_str = wg_pos_df.to_string(formatters={'Time': '{:.6f}'.format,
                                        'x': '{:.3f}'.format,
                                        'y': '{:.3f}'.format,
                                        'z': '{:.3f}'.format,
                                        'cov1': '{:.10e}'.format,
                                        'cov2': '{:.10e}'.format,
                                        'cov3': '{:.10e}'.format,
                                        'cov4': '{:.10e}'.format,
                                        'cov5': '{:.10e}'.format,
                                        'cov6': '{:.10e}'.format,
                                        'cov7': '{:.10e}'.format,
                                        'cov8': '{:.10e}'.format,
                                        'cov9': '{:.10e}'.format}, header=False, index=False)
        
        # Save to data folder
        
        # wg_df.to_csv("./data/wg_pos.txt", index=False, sep=' ', header=False)
        
        #open text file
        text_file = open('./data/wg_pos.txt', "w")
         
        #write string to file
        text_file.write(wg_df_str)
         
        #close file
        text_file.close()
        #    data_mat = np.append(data_mat, np.array([start_time + (time3[i]/1e6), x, y, z,
         #                        random.uniform(0.2e-3,0.8e-3),
          #                       random.uniform(0.1e-6,0.4e-6),
           #                      -random.uniform(0.8e-6,0.2e-5),
            #                     random.uniform(0.1e-6,0.4e-6),
             #                    random.uniform(0.2e-3,0.8e-3),
              #                   -random.uniform(0.1e-5,0.4e-5),
               #                  -random.uniform(0.8e-6,0.3e-5),
                #                 -random.uniform(0.1e-5,0.4e-5),
                 #                random.uniform(0.2e-3,0.8e-3)]))
        
        
        #%% Plot the shapes and the figures
        
        plt.figure(figsize=(10,10))
        plt.plot(long, lat, '.', markersize=5, label = 'Wave glider survey points')
        plt.plot(trans_latlong['lon'], trans_latlong['lat'], '^', markersize=12, label = 'Seafloor transponders')
        plt.plot(arr_center[1], arr_center[0], '.k', markersize=12, label = 'Array center')
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.legend(loc = 'upper right', fontsize = 15) #if need legend for the plots
        fig_name = str(shape) + '_' + str(rad) + 'm_' + str(int(np.size(x)/nrun_per_shape)) + 'pts_around_' + centered_around + '.png'
        plt.savefig('./Shape plots/'+fig_name)
        
        #%% save lat long of the data so that we can map it out in GIS software if wanted a nicer map
        
        # lat_long_df = pd.DataFrame([lat, long]).transpose()
        # lat_long_df.columns = ['lat', 'long']
        # shape_coord_file = str(shape) + '_' + str(rad) + 'm_' + str(num_pts) + 'p.csv' 
        # lat_long_df.to_csv('C:/Users/YXAVION/Documents/GitHub/Seafloor-geodesy-URECA-wave-glider/' + shape_coord_file ,sep = ',', index = False)
        
        
        #%% run command prompt gnatss from python
        # MUST CHANGE env.txt file to your own computer env
        
        mainPath = os.getcwd()
        command = 'gnatss run --extract-dist-center --extract-process-dataset'
        # distance limit and residual limit are set in the config.yaml file
        
        def parse_env_file(file_path):
            env_dict = {}
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        key, value = line.split('=', 1)
                        env_dict[key.strip()] = value.strip()
            return env_dict
        
        # Example usage:
        env_file_path = 'env.txt'
        env_dict = parse_env_file(env_file_path)
           
        from subprocess import Popen, PIPE, CalledProcessError
        
        with Popen(command, stdout=PIPE, bufsize=1, universal_newlines=True, env = env_dict, cwd = mainPath, shell = True) as p:
            for line in p.stdout:
                print(line, end='') # process line here
        
        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)
             
        # process = subprocess.run(command, cwd=mainPath, shell = True, env=env_dict, )
        # print(process.args)
        
        #%% save results in a folder for analysis
        
        src_folder = 'outputs'
        dst_folder = 'Saved outputs'
        new_dir = str(shape) + '_' + str(rad) + 'm_' + str(int(np.size(x)/nrun_per_shape)) + 'pts_around_' + centered_around
        
        if os.path.exists(src_folder):
            os.rename('outputs', new_dir)
        
        if os.path.exists(new_dir):
            try:
                shutil.move('./'+new_dir, './Saved outputs')
            except:
                print("Folder likely already exists in directory. Check again")
            
        
        
    