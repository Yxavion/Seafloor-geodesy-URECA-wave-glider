# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:51:18 2024

@author: YXAVION
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import pyproj
import scipy

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
        https://isis.astrogeology.usgs.gov/IsisSupport/index.php?topic=408.0
        and refers to (broken link)
        http://wegener.mechanik.tu-darmstadt.de/GMT-Help/Archiv/att-8710/Geodetic_py
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
    
    
#%% Parameters that are set, change this for diff models

# speed of sound in water
sv = 1500
depth = 2000

# let centre of the circle be 0,0,0

tran_pos1 = np.array([0, 2000, -2000])
tran_pos2 = np.array([-1720, -1021, -2000])
tran_pos3 = np.array([1720, -1021, -2000])

#%% Set up equation for the glider positions

# parametric equations
rad = depth

#The lower this value the higher quality the circle is with more points generated
stepSize = 0.1

#Generated vertices
positions = []

t = 0
while t < 2 * math.pi:
    positions.append([rad * math.sin(t), rad * math.cos(t), 0])
    t += stepSize
    
wg_pos = np.array(positions)


#%% calculate twtt for the trasnponders

# time(no) will be the twt of the transponders for every ping
obs_pts = len(wg_pos)

time1_clean = np.zeros(obs_pts)
time2_clean = np.zeros(obs_pts)
time3_clean = np.zeros(obs_pts)
 
# delay is in seconds
delay1 = 0.2
delay2 = 0.32
delay3 = 0.44

for i, wg in enumerate(wg_pos):
    time1_clean[i] = np.sqrt(np.inner(wg-tran_pos1, wg-tran_pos1))/sv
    time2_clean[i] = np.sqrt(np.inner(wg-tran_pos2, wg-tran_pos2))/sv
    time3_clean[i] = np.sqrt(np.inner(wg-tran_pos3, wg-tran_pos3))/sv
    
# add noise and delay to the signals 
time1 = (time1_clean + delay1 + np.random.normal(0, 0.0001, time1_clean.size))*10**6
time2 = (time2_clean + delay2 + np.random.normal(0, 0.0001, time1_clean.size))*10**6
time3 = (time3_clean + delay3 + np.random.normal(0, 0.0001, time1_clean.size))*10**6

time1 = time1.round()
time2 = time2.round()
time3 = time3.round()

# Write delay into text file with times
# insert time from random starting point

dates = pd.date_range(pd.Timestamp("26-JUN-23 00:00:07.00"),
                      freq="20S", periods = obs_pts)

dates = dates.strftime("%d-%b-%y %H:%M:%S.%f")

# Convert time array to dataframe

twt_df = pd.DataFrame([time1, time2, time3,dates])
twt_df = twt_df.T

twt_df.columns=['t1','t2','t3','dates']
twt_df = twt_df[['dates','t1','t2','t3']]
twt_df = twt_df.round()

twt_df.to_csv("twt.txt", index=False, sep='\t', header=False, 
              quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")



#%% Points of the wave glider, starting from the first point (0, 2000, 0)

# calculate angle from one point to the other

temp_lat, temp_long = vinc_pt(45.3023, -124.9656, 0, 2000) #initial point of the shape
# circle in this case


lat = np.zeros(obs_pts)
long = np.zeros(obs_pts)
x = np.zeros(obs_pts)
y = np.zeros(obs_pts)
z = np.zeros(obs_pts)

tempx, tempy, tempz = geodetic_to_cartesian(temp_lat, temp_long, 0)

# first observation point
lat[0] = temp_lat
long[0] = temp_long
x[0] = tempx
y[0] = tempy
z[0] = tempz

bearing_p2p = math.pi/2 # initial bearing if going east, clockwise

for i, wg in enumerate(wg_pos):

    if i < obs_pts-1:
    # calculate bearing of point to point. 
        bearing_p2p += stepSize
        bearing_dd = bearing_p2p*180/math.pi
        dist_moved = math.dist(wg_pos[i], wg_pos[i+1]) # dist moved from pt to pt
        
        temp_lat, temp_long = vinc_pt(lat[i], long[i], bearing_dd, dist_moved)
        lat[i+1] = temp_lat
        long[i+1] = temp_long
    
        x[i+1], y[i+1], z[i+1] = geodetic_to_cartesian(lat[i+1], long[i+1], 0)
    else:
        break


#%% Make the tranponder position file 
# needs time in j2000, XYZ, and covar 9 values

start_time = 741009607.0000000


data_mat = np.array([start_time, x[0], y[0], z[0], 0.7592545444E-03,
                     0.3218813907E-06,  -.3030846830E-05,
                     0.3218813907E-06,  0.7594897193E-03,
                     -.4333961609E-05,  -.3030846830E-05,
                     -.4333961609E-05,  0.7598263681E-03])

for i, wg in enumerate(wg_pos):

    data_mat = np.append(data_mat, np.array([start_time + (time1[i]/1e6), x[i], y[i], z[i],
                                             0.7592545444E-03,
                     0.3218813907E-06,  -.3030846830E-05,
                     0.3218813907E-06,  0.7594897193E-03,
                     -.4333961609E-05,  -.3030846830E-05,
                     -.4333961609E-05,  0.7598263681E-03]))
    
    data_mat = np.append(data_mat, np.array([start_time + (time2[i]/1e6), x[i], y[i], z[i],
                          0.7592545444E-03,
                          0.3218813907E-06,  -.3030846830E-05,
                          0.3218813907E-06,  0.7594897193E-03,
                          -.4333961609E-05,  -.3030846830E-05,
                          -.4333961609E-05,  0.7598263681E-03]))
    
    data_mat = np.append(data_mat, np.array([start_time + (time3[i]/1e6), x[i], y[i], z[i],
                           0.7592545444E-03,
                           0.3218813907E-06,  -.3030846830E-05,
                           0.3218813907E-06,  0.7594897193E-03,
                           -.4333961609E-05,  -.3030846830E-05,
                           -.4333961609E-05,  0.7598263681E-03]))
    
    start_time += 20
    
    if i == obs_pts - 1:
        break
    
    data_mat = np.append(data_mat, np.array([start_time, x[i+1], y[i+1], z[i+1],
                           0.7592545444E-03,
                           0.3218813907E-06,  -.3030846830E-05,
                           0.3218813907E-06,  0.7594897193E-03,
                           -.4333961609E-05,  -.3030846830E-05,
                           -.4333961609E-05,  0.7598263681E-03]))


data_mat_1 = np.reshape(data_mat, (int(len(data_mat)/13), 13))

# add uncertainties to geodetic cartesian coords
data_mat_1[:, 1:4] = data_mat_1[:, 1:4] + np.random.normal(
    0, 1.5, size=[len(data_mat_1[:, 1:4]), 3])


wg_pos_df = pd.DataFrame(data_mat_1)
#wg_pos_df['dumb column'] = '   '
#wg_pos_df = wg_pos_df[['dumb column',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

wg_pos_df.to_csv("wg_pos.txt", index=False, sep='\t', header=False, 
              quoting=csv.QUOTE_NONE, quotechar="", escapechar="\\")


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


