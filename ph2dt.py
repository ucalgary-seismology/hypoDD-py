#!/usr/bin/env python
import numpy as np
import os
import sys

from datetime import datetime
import misc_func as mf


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


def ph2dt():
	"""
	From HypoDD Version 1.3 -
	Author: Felix Waldhauser, felixw@ldeo.columbia.edu

	Purpose:
	Reads and filters absolute travel-time data from network catalogs
	to form travel-time data for pairs of earthquakes. Control file or
	interactive input of parameters to optimize linkage between events.

	For a user guide to ph2dt see USGS open-file report:
	   Waldhauser, F., HypoDD: A computer program to compute double-difference
	   earthquake locations,  U.S. Geol. Surv. open-file report , 01-113,
	   Menlo Park, California, 2001.

	Reads the input file specified om the command line (ph2dt ph2dt.inp)
	to get the input station and phase file names, and the numerical parameters.
	
	Writes the files (input files to hypoDD):
	dt.ct		the travel times for the common stations for close 
            	earthquake pairs
	event.dat	earthquake list in dd format
	event.sel	selected earthquake list in dd format (should be the same 
	            as event.dat)
	ph2dt.log	log file
	
	Most earthquake selection (dates, magnitudes, etc) happens before ph2dt 
	processes phase to differential time data.

	The standard input format (free format) is:
	#  yr  mn dy hr mn  sc      lat      lon      dep  mag  eh  ez  res  id
	sta  tt  wght pha
	e.g.:
	# 1997  6 13 13 44 37.33  49.2424 -123.6192   3.42 3.43 0.0 0.0 0.0  1
	NAB      4.871   1    P
	BIB      5.043   0.5  P
	SHB      7.043   1    S
	WPB      8.934   1    P

	See hypoDD user guide for a description of the parameters.
	"""

	PI = 3.141593
	KMPERDEG = 111.1949266
	MAXEV = 50

	# File with cuspids to select for
	fn9 = 'events.select'
	log = 'ph2dt.log'
	log = open(log,'w')
	datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	log.write('starting ph2dt %s \n' % datet)
	print('starting ph2dt %s' % datet)

	# Get input parameter file names:
	try:
		narguments = sys.argv
	except:
		print('User Enter Inputs:')
		inputfile = input('Inputfile location.  Default = "ph2dt.inp"')
		if not inputfile:
			inputfile = 'ph2dt.inp'

	if len(narguments)==1:
		print('User Enter Inputs:')
		inputfile = input('Inputfile location.  Default = "ph2dt.inp"')
		if not inputfile:
			inputfile = 'ph2dt.inp'
	elif len(narguments)==2:
		inputfile = narguments[0]
	else:
		raise Exception('Input file issues.')

	# Open input parameter file
	try:
		inputs = open(inputfile,'r')
		# Read input parameter file
		inputs = inputs.readlines()
		l = 1
		try:
			for line in inputs:
				if line[0:1] == '*' or line[1:2]=='*':
					continue
				else:
					if l==1:
						line = line.split('\n')
						statfile = line[0]
					if l==2:
						line = line.split('\n')
						phasefile = line[0]
					if l==3:
						line = line.split()
						line = list(filter(None,line)) 
						minwght = line[0]
						maxdist = line[1]
						maxoffset = line[2]
						mnb = line[3]
						limobs_pair = line[4]
						minobs_pair = line[5]
						maxobs_pair = line[6]
					l = l+1
		except:
			raise Exception('Error reading input file. Line %i \n %s' % (l,line))
	except:
		# Get modified (high-resolution station file name:)
		if not statfile:
			statfile = input('Station File [default="station.dat"')
			if not statfile:
				statfile = 'station.dat'
		try:
			stations = open(statfile,'r')
		except:
			raise Exception('Error opening station file.')
		# Get phase input file name:
		if not phasefile:
			phasefile = input('Phase File [default="phase.dat"')
			if not phasefile:
				phasefile = 'phase.dat'
		try:
			phases = open(phasefile,'r')
		except:
			raise Exception('Error opening phase file.')
		# Get max pick quality
		if not minwght:
			minwght = input('Min. pick weight/quality [default = 0.]:')
			if not minwght:
				minwght = 0.
		# Get max event-station distance
		if not maxdist:
			maxdist = input('Max. distance between event pair and station (km). [default = 200]:')
			if not maxdist:
				maxdist = 200.
		# Get max interevent offset (km) for which dtimes are calculated
		if not maxoffset:
			maxoffset = input('Max. epicentral separation (km) [default = 10]:')
			if not maxoffset:
				maxoffset = 10.
		# Get number of nearest neighbors
		if not mnb:
			mnb = input('Max. number of nearest neighbours per event [default = 10]:')
			if not mnb:
				mnb = 10
		# Get min number of dt-obs per pair so it counts as neighbour
		if not limobs_pair:
			limobs_pair = input('Min. number of links necessary to define a neighbour [default = 8]:')
			if not limobs_pair:
				limobs_pair = 8
		# Get max number of dt-obs per pair
		if not maxobs_pair:
			maxobs_pair = input('Max. number of links saved per pair [default = 50]:')
			if not maxobs_pair:
				maxobs_pair = 50

	# Open files:
	dts = open('dt.ct','w')
	events = open('event.dat','w')
	ev_sel = open('event.sel','w')
	stations = open(statfile,'r')
	phases = open(phasefile,'r')

	# Read icusps:
	ncusp = 0
	try:
		sel_ev = open(fn9,'r')
		try:
			sel_ev = sel_ev.readlines()
		except:
			raise Exception('Error reading event.select file.')
		ncusp = len(sel_ev)
		icusp = np.zeros(nev)
		for i,ev in enumerate(sel_ev):
			try:
				icusp[i] = int(ev)
			except:
				raise Exception('Error reading ID number. Line %i' % i)
	except:
		log.write('No events.select file. \n')
		print('No event.select file.')

	log.write('reading data... \n')
	print('reading data...')

	# Read stations
	try:
		stats = stations.readlines()
	except:
		raise Exception('Error reading station file.')
	nsta = len(stats)
	s_lab = np.empty(nsta,dtype='object')
	s_lat = np.zeros(nsta)
	s_lon = np.zeros(nsta)
	for i,sta in enumerate(stats):
		try:
			sta = sta.split()
			sta = list(filter(None,sta))
			s_lab[i] = str(sta[0])
			s_lat[i] = atoangle(sta[1])
			s_lon[i] = atoangle(sta[2])
		except:
			raise Exception('Error reading station file. Line %i' % i)
	log.write('> stations = %i \n' % nsta)
	print('> stations = %i' % nsta)
	stations.close()

	# Check for double stations:
	for i in range(0,nsta-1):
		for j in range(i+1,nsta):
			if s_lab[i] == s_lab[j]:
				log.write('This station is listed twice: %s \n' % s_lab[i])
				print('This station is listed twice: %s' % s_lab[i])
				sys.exit()

	# Read absolute network traveltimes
	try:
		phas = phases.readlines()
	except:
		raise Exception('Error reading phase file.')
	i = 0
	ii = 0
	npha = 0
	date = np.zeros(MAXEV)
	hr = np.zeros(MAXEV)
	minute = np.zeros(MAXEV)
	sec = np.zeros(MAXEV)
	lat = np.zeros(MAXEV)
	lon = np.zeros(MAXEV)
	depth = np.zeros(MAXEV)
	cuspid = np.zeros(MAXEV)
	nobs_ct = np.zeros(MAXEV)
	MAXOBS = MAXEV*nsta
	p_pha = np.empty((MAXEV,MAXOBS),dtype='object')
	p_sta = np.empty((MAXEV,MAXOBS),dtype='object')
	p_time = np.zeros((MAXEV,MAXOBS))
	p_wghtr = np.zeros((MAXEV,MAXOBS))
	count = 0
	for pha in phas:
		if pha[0]=='#':
			if count!=0:
				# Store Previous Event
				nobs_ct[i] = k
				itake = 1
				# Event Selection
				if ncusp > 0:
					itake = 0
					for k in range(0,ncusp):
						if cuspid[i] == icusp[k]:
							itake = 1
				# Write header to total event list
				events.write('%08i %08i %5.6f %5.6f %5.6f %1.2f %6.2f %6.2f %6.2f %8i \n' %
					         (date[i],rtime,lat[i],lon[i],depth[i],mag,herr,verr,res,cuspid[i]))
				# Keep event only if min_obs met
				if itake==1 and nobs_ct[i]>=int(minobs_pair):
					ev_sel.write('%08i %08i %5.6f %5.6f %5.6f %1.2f %6.2f %6.2f %6.2f %8i \n' %
					         	 (date[i],rtime,lat[i],lon[i],depth[i],mag,herr,verr,res,cuspid[i]))
					npha = npha+nobs_ct[i]
					i = i+1
			pha = pha.split()
			pha = list(filter(None,pha))
			yr = int(pha[1])
			mo = int(pha[2])
			dy = int(pha[3])
			date[i] = int(yr*10000 + mo*100 + dy)
			hr[i] = int(pha[4])
			minute[i] = int(pha[5])
			sec[i] = float(pha[6])
			rtime = int(hr[i]*1000000 + minute[i]*10000 + sec[i]*100)
			lat[i] = atoangle(pha[7])
			lon[i] = atoangle(pha[8])
			depth[i] = float(pha[9])
			mag = float(pha[10])
			herr = float(pha[11])
			verr = float(pha[12])
			res = float(pha[13])
			cuspid[i] = float(pha[14])
			ii = ii + 1
			k = 0  # Phase counter
			count = count+1
		else:
			pha = pha.split()
			pha = list(filter(None,pha))
			p_sta[i,k] = str(pha[0])
			p_time[i,k] = float(pha[1])
			p_wghtr[i,k] = float(pha[2])
			p_pha[i,k] = str(pha[3])
			if p_wghtr[i,k] > float(minwght) or p_wghtr[i,k] < 0:
				k = k+1

	# Store Last Event
	nobs_ct[i] = k
	itake = 1
	# Event Selection
	if ncusp > 0:
		itake = 0
		for k in range(0,ncusp):
			if cuspid[i] == icusp[k]:
				itake = 1
	# Write header to total event list
	events.write('%08i %08i %5.6f %5.6f %5.6f %1.2f %6.2f %6.2f %6.2f %8i' %
		         (date[i],rtime,lat[i],lon[i],depth[i],mag,herr,verr,res,cuspid[i]))
	# Keep event only if min_obs met
	if itake==1 and nobs_ct[i]>=int(minobs_pair):
		ev_sel.write('%08i %08i %5.6f %5.6f %5.6f %1.2f %6.2f %6.2f %6.2f %8i' %
		         	 (date[i],rtime,lat[i],lon[i],depth[i],mag,herr,verr,res,cuspid[i]))
		npha = npha+nobs_ct[i]
		i = i+1
	ii = ii + 1
	# Processing for all formats starts here:
	nev = i
	print('> events total = %i' % (ii-1))
	print('> events selected = %i' % nev)
	print('> phases = %i' % npha)
	log.write('> events total = %i \n' % (ii-1))
	log.write('> events selected = %i \n' % nev)
	log.write('> phases = %i \n' % npha)

	# Form dtimes
	print('Forming dtimes ...')
	log.write('Forming dtimes ... \n')
	log.write('Reporting missing stations (STA) and ourliers (STA,ID1,ID2,OFFSET (KM),T1,T2,T1-T2 \n')
	n1 = 0
	n2 = 0
	n3 = 0
	n4 = 0
	n5 = 0
	n6 = 0
	n7 = 0
	n8 = 0
	nerr = 0

	a_lab = np.empty(MAXOBS,dtype='object')
	a_pha = np.empty(MAXOBS,dtype='object')
	b_lab = np.empty(MAXOBS,dtype='object')
	b_pha = np.empty(MAXOBS,dtype='object')
	a_time1 = np.zeros(MAXOBS)
	a_time2 = np.zeros(MAXOBS)
	b_time1 = np.zeros(MAXOBS)
	b_time2 = np.zeros(MAXOBS)
	a_dist = np.zeros(MAXOBS)
	a_wtr = np.zeros(MAXOBS)
	b_wtr = np.zeros(MAXOBS)

	take = np.empty((nev,nev),dtype='object')
	for i in range(0,nev):
		for j in range(0,nev):
			take[i,j] = '1'

	ipair = 1
	ipair_str = 1
	avoff = 0
	avoff_str = 0
	maxoff_str = 0

	aoffs = np.zeros(nev)
	for i in range(0,nev):
		# Find nearest neighbor
		for j in range(0,nev):
			dlat = lat[i] - lat[j]
			dlon = lon[i] - lon[j]
			x = dlat*KMPERDEG
			y = dlon*(np.cos(lat[i]*PI/180.)*KMPERDEG)
			z = depth[i]-depth[j]
			aoffs[j] = np.sqrt(x*x + y*y + z*z)

		aoffs[i] = 99999
		#indx = indexx(nev,aoffs)
		indx = np.argsort(aoffs)

		inb = 0
		nobs = 0
		for m in range(0,nev-1): # same event last
			if inb >= int(mnb):
				break # next event
			k = indx[m] # nearest event first
			if take[k,i] == '0': # already selected as a strong neighbor
				inb = inb+1
				continue
			elif take[k,i] == '9': # weak neighbor
				continue

			n1 = n1+1

			# Check max interevnt offset:
			if aoffs[indx[m]] > float(maxoffset):
				break

			# Search for common stations/phases:
			iobs = 0
			iobsP = 0
			iimp = 0 # obs needs to be included regardless of weight or dist
			for j in range(0,int(nobs_ct[i])):
				n5break=0
				for l in range(0,int(nobs_ct[k])):
					if p_sta[i,j] == p_sta[k,l] and p_pha[i,j] == p_pha[k,l]:
						if p_pha[i,j] == 'P':
							n3 = n3+1
						if p_pha[i,j] == 'S':
							n6 = n6+1

						# Check for station label in station file:
						okbreak=0
						for ii in range(0,nsta):
							ok = 0
							if p_sta[i,j] == s_lab[ii]:
								ok = 1
							# Select for station-pair centroid distance
							if ok==1:
								alat = s_lat[ii]
								alon = s_lon[ii]
								blat = (lat[i] + lat[k])/2.
								blon = (lon[i] + lon[k])/2.
								delt,dist,az = mf.delaz(alat,alon,blat,blon)

								# Delete far away stations
								if dist>int(maxdist):
									n5 = n5+1
									n5break=1
									break
								okbreak=1
								break

						if n5break==1:
							break
						if okbreak==0:
							n4 = n4 + 1
						ista = ii
						# Get avg. weight
						wtr = (np.abs(p_wghtr[i,j]) + abs(p_wghtr[k,l]))/2.

						# Remove outliers above the separation-delaytime line:
						if p_pha[i,j] == 'P':
							vel = 4.
						if p_pha[i,j] == 'S':
							vel = 2.3
						if np.abs(p_time[i,j] - p_time[k,l]) > (aoffs[indx[m]]/vel + 0.5):
							log.write('Outlier: %s %08i %08i %3.5f %3.5f %3.5f %3.5f \n' %
									  (p_sta[i,j],cuspid[i],cuspid[k],aoffs[indx[m]],p_time[i,j],p_time[k,l],p_time[i,j]-p_time[k,l]))
							nerr = nerr+1
							break

						if p_pha[i,j] == 'P':
							iobsP = iobsP+1
						nobs = nobs+1
						a_lab[iobs] = s_lab[ista]
						a_time1[iobs] = p_time[i,j]
						a_time2[iobs] = p_time[k,l]
						a_wtr[iobs] = wtr
						a_dist[iobs] = dist
						a_pha[iobs] = p_pha[i,j]
						if p_wghtr[i,j] < 0. or p_wghtr[k,l] < 0:
							a_dist[iobs] = 0. # Set dist to 0 so it's selected first
							iimp = iimp+1
						iobs = iobs+1
						break

			itmp = iobs
			if iobs>int(maxobs_pair):
				itmp = min(int(maxobs_pair) + iimp, iobs)
			if iobs>=int(minobs_pair):
				if iobs>1:
					iindx = np.argsort(a_dist[0:iobs])
					for kk in range(0,iobs):
						b_lab[kk] = a_lab[iindx[kk]]
						b_time1[kk] = a_time1[iindx[kk]]
						b_time2[kk] = a_time2[iindx[kk]]
						b_wtr[kk] = a_wtr[iindx[kk]]
						b_pha[kk] = a_pha[iindx[kk]]
				else:
					b_lab[0] = a_lab[0]
					b_time1[0] = a_time1[0]
					b_time2[0] = a_time2[0]
					b_wtr[0] = a_wtr[0]
					b_pha[0] = a_pha[0]

				# Write out delay times
				dts.write('# %8i %8i \n' % (cuspid[i],cuspid[k]))
				for kk in range(0,itmp):
					dts.write('%s %7.3f %7.3f %6.4f %s \n' % (b_lab[kk],b_time1[kk],b_time2[kk],b_wtr[kk],b_pha[kk]))
					if b_pha[kk] == 'P':
						n7 = n7+1
					if b_pha[kk] == 'S':
						n8 = n8+1
				avoff = avoff + aoffs[k]
				ipair = ipair + 1

			if iobs>=int(limobs_pair): # select as strong neighbor
				take[i,k] = '0'
				inb = inb+1
				ipair_str = ipair_str + 1
				avoff_str = avoff_str + aoffs[k]
				if aoffs[k] > maxoff_str:
					maxoff_str = aoffs[k]
			else: # weak neighbor
				take[i,k] = '9'
		
		if inb<int(mnb):
			n2 = n2+1

	npair = ipair-1
	avoff = avoff/npair
	avoff_str = avoff_str/(ipair_str-1)

	print('> P-phase pairs total = ',n3)
	print('> S-phase pairs total = ',n6)
	print('> outliers = ',nerr,' (',nerr*100/(n3+n6),'%)')
	print('> phases at stations not in station list = ',n4)
	print('> phases at distances larger than MAXDIST = ',n5)
	if n3>0:
		print('> P-phase pairs selected = ',n7,'(',n7*100/n3,'%)')
	if n6>0:
		print('> S-phase pairs selected = ',n8,'(',n8*100/n6,'%)')
	print('> weakly linked events = ',n2,'(',n2*100/nev,'%)')
	print('> linked event pairs = ',ipair)
	print('> average links per pair = ',int((n7+n8)/ipair))
	print('> average offset (km) betw. linked events = ',avoff)
	print('> average offset (km) betw. strongly linked events = ',avoff_str)
	print('> maximum offset (km) betw. strongly linked events = ',maxoff_str)
	log.write('> P-phase pairs total = %i \n' % n3)
	log.write('> S-phase pairs total = %i \n' % n6)
	log.write('> outliers = %i (%3.2f percent) \n' % (nerr, nerr*100/(n3+n6)))
	log.write('> phases at stations not in station list = %i \n' % n4)
	log.write('> phases at distances larger than MAXDIST = %i \n' % n5)
	if n3>0:
		log.write('> P-phase pairs selected = %i (%3.2f percent) \n' % (n7,n7*100/n3))
	if n6>0:
		log.write('> S-phase pairs selected = %i (%3.2f percent) \n' % (n8,n8*100/n6))
	log.write('> weakly linked events = %i (%3.2f percent) \n' % (n2,n2*100/nev))
	log.write('> linked event pairs %i \n' % ipair)
	log.write('> average offset (km) betw. linked events = %4.5f \n' % avoff)
	log.write('> average offset (km) betw. strongly linked events = %4.5f \n' % avoff_str)
	log.write('> maximum offset (km) betw. strongly linked events = %4.5f \n' % maxoff_str)

	phases.close()
	dts.close()
	events.close()
	ev_sel.close()

	datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	log.write('Done %s \n' % datet)
	print('Done %s' % datet)

	log.close()

	print('Output files: dt.ct; event.dat; event.sel; ph2dt.log')
	print('ph2dt parameters were: ')
	print('(minwght,maxdist,maxsep,maxngh,minlnk,minobs,maxobs)')
	print(minwght,maxdist,maxoffset,mnb,limobs_pair,minobs_pair,maxobs_pair)



ph2dt()









