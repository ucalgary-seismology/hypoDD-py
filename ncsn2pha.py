#!/usr/bin/env python
# This script converts functionality of the ncsn2pha fortran function to full python scripting

"""
ncsn2pha converts Hypoinverse Y2000 archive phase format into the phase
format wanted by the ph2dt program, which is part of the hypoDD process.

This program must be run with the input and output files specified 
on the command line.

Both input and output files are organized with an earthquake location 
header, then a list of phases.

Output file can be read either free format or fixed column format.

The NCSN convention of naming verical stations is followed when selecting only
vertical stations for P phases.

The option of converting HVO archive files with HVO station naming (and finding
vertical stations for P phases) is invoked with "h" as the third argument:
ncsn2pha inputfile outputfile [h]

"""

import sys


# Get input file name
narguments = sys.argv
if narguments < 3:
	raise Exception('Not enough arguments. Use form \'ncsn2pha inputfile outputfile\'')
try:
	inputfile = open(sys.argv[1],'r')
except:
	raise Exception('Error opening input file.')
try:
	outputfile = open(sys.argv[2],'w')
except:
	raise Exception('Error opening output file.')

# Make NCSN station name assumptoions, unless "h" is given as the 3rd argument
# if h=True then we make HVO station assumptions
hvert = False
if narguments > 3 and sys.argv[3]=='h':
	hvert = True


lines = inputfile.readlines()
# Read header
for line in lines:
	if lines[0] == '$' # Skip shadow lines
		continue
	if line[0:2] == '19' or line[1:2] == '20':
		line[0:8] = int(date)
		line[4:6] = int(mo)
		line[6:8] = int(dy)
		line[8:10] = int(hr)
		line[10:12] = int(mins)
		line[12:16] = float(sec)
		line[16:18] = float(deglat)
		line[19:23] = float(minlat)
		line[23:26] = float(deglon)
		line[27:31] = float(minlon)
		line[31:36] = float(depth)
		line[147:150] = float(mag)
		line[48:52] = float(res)
		line[85:89] = float(herr)
		line[89:93] = float(verr)
		line[136:146] = int(cuspid)
	else:
		line[0:6] = int(date)
		date = date+19000000
		line[2:4] = int(mo)
		line[4:6] = int(dy)
		line[6:8] = int(hr)
		line[8:10] = int(mins)
		line[10:14] = float(sec)
		line[14:16] = float(deglat)
		line[17:21] = float(minlat)
		line[21:24] = float(deglon)
		line[25:29] = float(minlon)
		line[29:34] = float(depth)
		line[67:69] = float(mag)
		line[45:49] = float(res)
		line[80:84] = float(herr)
		line[84:88] = float(verr)
		line[128:138] = int(cuspid)

	yr = date/10000
	lat = deglat + minlat/60
	rtime = hr*1000000 + mins*10000 + sec*100
	sec = int(sec*100)

	# Write to earthquake location file
	outputfile.write('# %02i%02i%02i %02i%02i%04i %4.4f %4.4f %5.2f %4.2f %4.2f %4.2f %010i'
		              % (yr,mo,dy,hr,mins,sec,lat,lon,depth,mag,herr,verr,res,cuspid))

	# Read data lines and get time
	k = 1



