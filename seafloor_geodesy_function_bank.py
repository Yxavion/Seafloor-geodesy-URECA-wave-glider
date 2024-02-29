#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:15:19 2024

@author: masonperry
"""
## Function bank for seafloor geodesy
import numpy as np
import pandas as pd
import netCDF4 as nc
import time

## Function to take ecef coordinates and move them to geodetic
def ecef_to_geodetic(x, y, z):
    # WGS84 parameters
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening

    e_squared = 2*f - f**2  # eccentricity squared

    # Calculate longitude (lambda)
    lon = np.arctan2(y, x)

    # Calculate latitude (phi)
    p = np.sqrt(x**2 + y**2)
    phi = np.arctan2(z, p * (1 - e_squared))

    # Iterative calculation for altitude (h)
    phi_prev = np.zeros_like(phi)
    while np.any(np.abs(phi - phi_prev) > 1e-10):
        N = a / np.sqrt(1 - e_squared * np.sin(phi)**2)
        h = p / np.cos(phi) - N
        phi_prev = phi
        phi = np.arctan2(z, p * (1 - e_squared * N / (N + h)))

    # Convert latitude and longitude to degrees
    lat = np.degrees(phi)
    lon = np.degrees(lon)
    out = pd.DataFrame([lat,lon,h]).T
    out.columns = ['lat','long','elev']
    return out

## Function to take geodetic coordinates and transform to earth centered
## Note: to use another datum than WGS84, you'll need to change a and f
def geodetic_to_ecef(lon, lat, elevation):
    # WGS84 parameters
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening

    e_squared = 2*f - f**2  # eccentricity squared

    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate the prime vertical radius of curvature (N)
    N = a / np.sqrt(1 - e_squared * np.sin(lat_rad)**2)

    # Calculate Cartesian coordinates
    x = (N + elevation) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + elevation) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e_squared) + elevation) * np.sin(lat_rad)
    out = pd.DataFrame([x,y,z]).T
    out.columns = ['X','Y','Z']
    return out

def add_dec_year_from_datetime(df):
    def toYearFraction(date):
        def sinceEpoch(date):
            return time.mktime(date.timetuple())
        s = sinceEpoch
        year = date.year
        startOfYear = dt(year=year,month=1,day=1)
        startOfNextYear = dt(year=year+1,month=1,day=1)
        yearElapsed = s(date) - s(startOfYear)
        yearDuration = s(startOfNextYear)-s(startOfYear)
        frac = yearElapsed/yearDuration
        return date.year+frac
    df["dec_year"]=df["datetime"].apply(toYearFraction).astype('float')
    return df
from scipy.optimize import curve_fit

## Function to read gnatss netcdf output files
def read_gnatss_nc_file(filename,output_info='no'):
    """
    Function to read gnatss output nc files and convert data into a form easily useable in python

    Parameters
    ----------
    filename : Filename with path of the nc file output from gnatss processing.

    Returns 
    -------
    out_dic: A dictionary with the variables in the netcdf file. Note: this may get more complicated and is subject to errors if loading more complex netcdf files

    """
    file = nc.Dataset(filename,'r')
    variables = list(file.variables)
    out_dic = {}
    for i in variables:
        out_dic[i] = np.array(file[i])

    # transponder = list(file['transponder'][:])
    # coords = list(file['coords'][:])
    # transponders_xyz = np.array(file['transponders_xyz'][:])
    # delta_xyz = np.array(file['delta_xyz'][:])
    # sigma_xyz = np.array(file['sigma_xyz'][:])
    # delta_enu = np.array(file['delta_enu'][:])
    # sigma_enu = np.array(file['sigma_enu'][:])
    # transponders_lla = np.array(file['transponders_lla'][:])
    # rms_residual = np.array(file['rms_residual'][:])
    # error_factor = np.array(file['error_factor'][:])
    # iteration = np.array(file['iteration'][:])
    file.close()
    
    #return transponder,coords,transponders_xyz,delta_xyz,sigma_xyz,delta_enu,sigma_enu, transponders_lla,rms_residual,error_factor,iteration
    return out_dic