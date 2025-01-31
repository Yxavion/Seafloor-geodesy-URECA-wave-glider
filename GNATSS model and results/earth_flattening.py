# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:08:30 2025

@author: Hari Vishnu
"""
import numpy as np
    
def eccentricity(a,b):
    """
    Parameters
    ----------
    a : semi-major axis of ellipsoid Earth. Should be default value unless Earth changes
    b : semi-minor axis of ellipsoid Earth. Should be default value unless Earth changes

    Returns
    -------
    e : eccentricity

    """
    e = np.sqrt(a**2-b**2)/a #eq 3 in Chadwell and Sweeney
    return e

def R_G(phi, a = 6378137.000, b = 6356752.3):
    """
    Implements equations in "Acoustic Ray-Trace Equations for Seafloor Geodesy" by Chadwell and Sweeney
    
    to obtain the radius of the spherical Earth model which is the Gaussian mean radius of curvature, 
    that can then be used to do an Earth-flattening transform for seafloor geodesy.
    
    phi: latitude (radians)
    a : semi-major (equatorial) axis of ellipsoid Earth. Should be default value unless Earth changes (m)
    b : semi-minor (polar) axis of ellipsoid Earth. Should be default value unless Earth changes (m)
    
    Returns
    -------
    R_alpha: float

    """
    N_phi = a/np.sqrt(1-e**2*np.sin(phi)**2) #radius of curvature in prime vertical eq 2
    M_lambda = a*(1-e**2)/np.pow(1-e**2*np.sin(phi)**2,1.5) #Meridional radius of curvature, eq 4
    R = np.sqrt(N_phi*M_lambda) #eq 45
    return R
    
def R_alpha(phi,H,alpha, a = 6378137.000, b = 6356752.3):
    """
    Implements equations in "Acoustic Ray-Trace Equations for Seafloor Geodesy" by Chadwell and Sweeney
    
    to obtain the radius of the spherical Earth model tangential along the particular azimuth
    being studied, that can then be used to do an Earth-flattening transform for seafloor geodesy.
    
    phi: latitude (radians)
    H: orthometric height = h-N where it is height above the geoid (m)
    alpha: azimuth of the ray from transducer to receiver, clockwise with respect to North (radians)
    a : semi-major (equatorial) axis of ellipsoid Earth. Should be default value unless Earth changes (m)
    b : semi-minor (polar) axis of ellipsoid Earth. Should be default value unless Earth changes (m)
    
    Returns
    -------
    R_alpha: float

    """
    e = eccentricity(a, b)
    N_phi = a/np.sqrt(1-e**2*np.sin(phi)**2) #radius of curvature in prime vertical eq 2
    M_lambda = a*(1-e**2)/np.pow(1-e**2*np.sin(phi)**2,1.5) #Meridional radius of curvature, eq 4. Is this wrong, should it be lambdal instead of phi?
    
    R = 1/(np.cos(alpha)**2/M_lambda + np.sin(alpha)**2/N_phi)#eq 46
    return R

def heights_transformed(h, R):
    """
    Generate Earth-flattening transform for new heights in equivalent planar surface for spherical model

    Parameters
    ----------
    h : float or array of float (m)
        heights in spherical model
    R : float or array of float (m)
        Radius of spherical model Earth to use. R_alpha is the best approximation as per Chadwell, though R_G also works.

    Returns 
    -------
    float or array of float (m)
        heights transformed.

    """
    return np.multiply(R, np.log(np.divide(R, R-h)))

def SSP_transformed(c, h, R):
    """
    Generate Earth-flattening transform for new SSP in equivalent planar surface for spherical model

    Parameters
    ----------
    c : float or array of float (m/s)
        SSP in spherical model
    h : float or array of float (m)
        heights in spherical model
    R : float or array of float (m)
        Radius of spherical model Earth to use. R_alpha is the best approximation as per Chadwell, though R_G also works.

    Returns 
    -------
    float or array of float (m)
        SSP transformed.

    """
    return np.multiply(c, np.divide(R, R-h))

def eccentricity_r(reverse_flattening = 298.257222101)
    """
    Compute eccentricity based on code segment I found in GNATSS: https://github.com/seafloor-geodesy/gnatss/blob/3281937815640f07078283011f852261945e3f7c/src/gnatss/configs/solver.py#L29
    Parameters
    ----------
    reverse_flattening : Should be 298.257222101 unless Earth changes

    Returns
    -------
    e : eccentricity

    """
    return 2.0 / reverse_flattening - (1.0 / reverse_flattening) ** 2.0
