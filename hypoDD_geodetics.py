import numpy as np
import line_profiler
import atexit
from numba import jit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

# Global coordinate system variables
rearth = float(0.)
ellip = float(0.)
rlatc = float(0.)
rad = float(0.)
olat = float(0.)
olon = float(0.)
aa = float(0.)
bb = float(0.)
bc = float(0.)
sint = float(0.)
cost = float(0.)
rotate = float(0.)
icoordsystem = int(0)


#@profile
def setorg(orlat,orlon,rota,ifil):
	# Set up cartesian coordinate system by short distance conversion
	# Unrotated coordinate system with pos. x-axis toward WEST
	# and pos. y-axis toward NORTH
	# pos. z-axis toward EARTH'S CENTER

	global rearth, ellip, rlatc, rad
	global olat, olon, aa, bb, bc
	global sint, cost, rotate
	global icoordsystem

	# if orlat or orlon are both set to zero, the Swiss Cartesian 
	# coordinate system will be used (this system cannot be rotated)
	rad = 0.017453292

	if orlat==0. and orlon==0.:
		olat = 46.95240 # BERN North
		olon = -7.439583 # BERN West
		rotate = 0.
	else:
		olat = orlat
		olon = orlon
		rotate = rota*rad

	olat = olat*60. # Minutes N
	olon = olon*60. # Minutes 

	# New Ellipsoid for Whol Earth: WGS72
	# Also set rlatc according to origin
	rearth = 6378.135
	ellip = 298.26 # Flattening

	# Calculate Rlatc: Conversion from geographical lat to geocentrical lat
	phi = olat*rad/60.					# phi = geogr. lat
	beta = phi-np.sin(phi*2.)/ellip 	# beta = geoc. lat
	rlatc = np.tan(beta)/np.tan(phi)

	# Write
	if ifil>0:
		ifile = open(ifil,'w')
		ifile.write('SHORT DISTANCE CONVERSION on ELLIPSOID of')
		ifile.write('WORLD GEODETIC SYSTEM 1972 (WGS72) \n')
		ifile.write('======================================')
		ifile.write('======================================\n \n')
		ifile.write('(Radius at equator (rearth) = %10.5f km) \n' % rearth)
		ifile.write('(1./(ellipticity) = %10.3f \n \n' % ellip)
		ifile.write('Origin of cartesian coordinates [degrees]: \n\n')
		if orlat==0. and orlon==0.:
			ifile.write('SWISS COORDINATE SYSTEM \n\n')
			ifile.write('(Origin = city of BERN, Switzerland)\n\n')
			ifile.write('no rotation of grid, pos. y-axis toward N \n')
			ifile.write('                     pos. x-axis toward E \n\n')
		else:
			ifile.write('( %12.7f N     %12.7f W )\n\n' % (olat/60.,olon/60.))
			ifile.write(' without rotation of grid, \n')
			ifile.write('               pos. x-axis toward W \n')
			ifile.write('               pos. y-axis toward N \n\n')
			ifile.write(' Rotation of y-axis from North anticlockwise \n')
			ifile.write(' with pos. angle given in degrees \n\n')
			if rota>=0:
				ifile.write(' Rotation of grid anticlockwise by \n')
				ifile.write(' %10.5f degrees \n\n' % rota)
			else:
				ifile.write(' Rotation of grid clockwise by \n')
				arota = -1.*rota
				ifile.write(' %10.5f degrees \n\n' % arota)

	# Calculate aa and bb
	# Length of one minute of lat and lon in km
	lat1 = np.arctan(rlatc*np.tan(olat*rad/60.))	# geoc. lat for OLAT
	lat2 = np.arctan(rlatc*np.tan((olat+1.)*rad/60.))	# geoc. lat for (OLAT + 1 min)
	dela = lat2-lat1
	r = rearth*(1. - (np.sin(lat1)**2)/ellip)		# spherical radius for lat=olat
	aa = r*dela		# aa = 1 min geogr. lat
	delb = np.arccos(np.sin(lat1)**2 + np.cos(rad/60.)*np.cos(lat1)**2)
	bc = r*delb 	# bc = 1 min geogr. lon
	bb = r*delb/np.cos(lat1)
	if ifil>0:
		ifile.write('( Radius of sphere of OLAT = %10.3f km )\n\n' % r)
		ifile.write('Conversion of GEOGRAPHICAL LATITUDE to GEOCENTRICAL LATITUDE \n')
		ifile.write('RLATC = TAN(GEOCENTR.LAT) / TAN(GEOGRAPH.LAT) \n')
		ifile.write('( RLATC = %12.8f ) \n\n' % rlatc)
		ifile.write('Short distance conversions: \n')
		ifile.write('one min lat ~ %7.4f km\n' % aa)
		ifile.write('one min lon ~ %7.4 km\n' % bc)
		ifile.close()

	# Convert coordinates with rotation cosines (stored in common)
	sint = np.sin(rotate)
	cost = np.cos(rotate)

	return
	

#@profile
#@jit
def dist(xlat,xlon):
	# Convert latitude and longitude to kilometers relative to
	# center of coordinates by short distance conversion.

	global rearth, ellip, rlatc, rad
	global olat, olon, aa, bb, bc
	global sint, cost, rotate
	global icoordsystem

	# Set up short distance conversion by subr. SETORG
	q = 60.*xlat - olat
	yp = q+olat
	lat1 = np.arctan(rlatc*np.tan(rad*yp/60.))
	lat2 = np.arctan(rlatc*np.tan(rad*olat/60.))
	lat3 = (lat2+lat1)/2.
	xx = 60.*xlon-olon
	q = q*aa
	xx = xx*bb*np.cos(lat3)
	if rotate!=0:
		yp = cost*q*sint*xx
		xx = cost*xx-sint*q
		q=yp

	xkm=xx
	ykm=q

	return xkm,ykm


#@profile
#@jit
def redist(xkm,ykm):
	# Convert from local Cartesian coordinates to lat and lon

	global rearth, ellip, rlatc, rad
	global olat, olon, aa, bb, bc
	global sint, cost, rotate
	global icoordsystem

	xx = xkm
	yy = ykm

	# Rotate coordinates anticlockwise back
	y = yy*cost-xx*sint
	x = yy*sint+xx*cost
	if abs(aa)>0.0000001:
		q = y/aa
		lat = (q+olat)/60.
		xlat = q+olat - 60.*lat
		yp = 60.*lat + xlat
		lat1 = np.arctan(rlatc*np.tan(yp*rad/60.))
		lat2 = np.arctan(rlatc*np.tan(olat*rad/60.))
		lat3 = (lat1+lat2)/2.
		clat1 = np.cos(lat3)
		bcl = bb*clat1
		if abs(bcl)>0.000001:
			p = x/(bb*clat1)
			lon = (p+olon)/60.
			xlon = p+olon - 60.*lon
			xlat = lat + xlat/60.
			xlon = lon + xlon/60.
			return xlat,xlon
	print('subr. redist: \n')
	print('aa = %10.5f \n' % aa)
	print('bb = %10.5f \n' % bb)
	print('cos(lat1) = %10.5f \n' % clat1)

	raise Exception('division by zero run stops here')


#@profile
#@jit
def sdc2(xlat,xlon,i):
	# Convert coordinates of a point by short distance conversion
	if i!=1 and i!=-1:
		raise Exception('SDC: Specify conversion')
	if i==1:
		x,y, = redist(xlat,xlon)
	if i==-1:
		x,y = dist(xlat,xlon)

	return x,y

