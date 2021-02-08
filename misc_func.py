#!/usr/bin/env python
import numpy as np
import os
import sys

# Functions shared by multiple commands in hypoDD v1.3

def atoangle(locstr):
    """
    Convert string from form "degrees:minutes:seconds"
    to angle if needed. Otherwise just return float of angle.
    ###########
    PARAMETERS:
    locstr (str) ---- Lat or lon string from file
    ###########
    RETURNS:
    loc (float) ---- Lat or lon in angle form
    ###########
    """
    if ':' in locstr:
        loc = locstr.split(':')
        loc = float(loc[0]) + (float(loc[1]) + float(loc[2])/60.)/60.
    else:
        loc = float(locstr)

    return loc

def vmodel(nl, v, top, depth):
    """
    Extract needed information from the velocity model
    ###########
    PARAMETERS:
    nl (int) ---- Number of layers in velocity model
    v[nl] (float array) ---- Velocity in each layer
    top[nl] (float array) ---- Fepth to top of layer
    depth (float) ---- Focal depth of source in km
    ###########
    RETURNS:
    vsq[nl] (float array) ---- Squared velocities
    thk[nl] (float array) ---- Thicknesses of layers
    jl (int) ---- Event layer
    tkj (float) ---- Depth of event in event layer
    ###########
    """
    vsq = np.zeros(nl)
    thk = np.zeros(nl)
    for i in range(0,nl):
        if i==nl:
            thk[i] = 5000.
        vsq[i] = v[i]*v[i]
        thk[i] = top[i+1]-top[i]
    
    jl = nl-1
    for i in reversed(range(0,nl)):
        if depth <= top[i]:
            jl = i-1
    
    # Depth from top of layer to source
    tkj = depth-top[jl]

    return vsq,thk,jl,tkj
    

def delaz(alat, alon, blat, blon):
    """
    This function computes distance and azimuth on a sphere
    ###########
    PARAMETERS:
    alat (float) ---- Latitude of first point
    alon (float) ---- Longitude of first point
    blat (float) ---- Latitude of second point
    blon (float) ---- Longitude of second point
    ###########
    RETURNS:
    delt (float) ---- Sentral andle (degrees)
    dist (float) ---- Distance (km)
    az (float) ---- Azimuth from a to b (degrees)
    ###########
    """
    # Variables declared in delaz2.f (original fortran)
    # Kept consistent to retain same values no
    pi2 = 1.570796
    rad = 1.745329e-2
    flat = .993231
    # Convert to radians
    alatr = alat*rad
    alonr = alon*rad
    blatr = blat*rad
    blonr = blon*rad
    # Convert latitudes to geocentric colatitudes
    tana = flat*np.tan(alatr)
    geoa = np.arctan(tana)
    acol = pi2 - geoa
    tanb = flat*np.tan(blatr)
    geob = np.arctan(tanb)
    bcol = pi2 - geob
    # Calculate delta
    diflon = blonr-alonr
    cosdel = np.sin(acol)*np.sin(bcol)*np.cos(diflon) + np.cos(acol)*np.cos(bcol)
    delr = np.arccos(cosdel)
    # Calculate azimuth from a to b
    top = np.sin(diflon)
    den = (np.sin(acol)/np.tan(bcol)) - np.cos(diflon)*np.cos(acol)
    azr = np.arctan2(top,den)
    # Convert to degrees
    delt = delr/rad
    az = azr/rad
    if az < 0.0:
        az = 360.+az
    # Compute distance in km
    #print('Alatr %f' % alatr)
    #print('Blatr %f' % blatr)
    colat = pi2 - (alatr+blatr)/2.
    #print('Colat: %f' % colat)
    # The equatorial radius of the Earth is 6378.137 km (IUGG value)
    # The mean equatorial radius from Bott 1982 is 6378.140 km
    radius = 6378.140*(1.0+3.37853e-3*(1./3.-((np.cos(colat))**2)))
    dist = delr*radius

    return delt, dist, az