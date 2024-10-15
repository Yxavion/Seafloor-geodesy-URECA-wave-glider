# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:18:51 2024

@author: YXAVION
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

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

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

#%% two way travel time from bellhop
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

#%% Shape create file with the time element




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

#%% Parameters set for the different models

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
            # function called shape_create
            # the loop will be for multiple iterations of the same shape to add some noise and test the shape thoroughly
            # add with time element
            
        
        
        
        
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
        
        # convert to ms ##############################
        time1 = (time1_clean + delay1 + np.random.normal(0, 0.00001, time1_clean.size))*10**6
        time2 = (time2_clean + delay2 + np.random.normal(0, 0.00001, time1_clean.size))*10**6
        time3 = (time3_clean + delay3 + np.random.normal(0, 0.00001, time1_clean.size))*10**6
        
                
            

