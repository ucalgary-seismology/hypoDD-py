import numpy as np
import os
from datetime import datetime
import misc_func as mf
import rt_functions as rt
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


# HypoDD data input/output functions
#@profile
def getinp(log,fn_inp):
    """
    Get input parameters
    ##########
    Inputs:
    log (file obj): opened file location for log file
    fn_inp (str):   str file location for input file
    ##########
    Outputs:
    retlist (list): list object containing all input parameters 
                    from inp file
    """
    # Open input file
    inputfile = open(fn_inp,'r')
    ncusp = 0
    niter = 0 # number of iteration blocks
    l = 0
    ii = 0
    icusp = []
    # Loop to read each parameter line, skipping comments
    inputs = inputfile.readlines()
    for line in inputs:
        if line[0:1] == '*' or line[1:2] == '*':
            continue
        line = line.split()
        line = list(filter(None,line))
        if l==0:
            fn_cc = str(line[0])
        if l==1:
            fn_ct = str(line[0])
        if l==2:
            fn_eve = str(line[0])
        if l==3:
            fn_sta = str(line[0])
        if l==4:
            fn_loc = str(line[0])
        if l==5:
            fn_reloc = str(line[0])
        if l==6:
            fn_stares = str(line[0])
        if l==7:
            fn_res = str(line[0])
        if l==8:
            fn_srcpar = str(line[0])
        if l==9:
            idata = int(line[0])
            iphase = int(line[1])
            maxdist = float(line[2])
        if l==10:
            minobs_cc = int(line[0])
            minobs_ct = int(line[1])
        if l==11:
            istart = int(line[0])
            isolv = int(line[1])
            niter = int(line[2])
            # Read iteration instructions
            aiter = np.zeros(niter)
            awt_ccp = np.zeros(niter)
            awt_ccs = np.zeros(niter)
            amaxres_cross = np.zeros(niter)
            amaxdcc = np.zeros(niter)
            awt_ctp = np.zeros(niter)
            awt_cts = np.zeros(niter)
            amaxres_net = np.zeros(niter)
            amaxdct = np.zeros(niter)
            adamp = np.zeros(niter)
        if l >= 12 and l <= 11+niter:
            i = l-12
            aiter[i] = int(line[0])
            awt_ccp[i] = float(line[1])
            awt_ccs[i] = float(line[2])
            amaxres_cross[i] = float(line[3])
            amaxdcc[i] = float(line[4])
            awt_ctp[i] = float(line[5])
            awt_cts[i] = float(line[6])
            amaxres_net[i] = float(line[7])
            amaxdct[i] = float(line[8])
            adamp[i] = int(line[9])
        if l==12+niter:
            mod_nl = int(line[0])
            mod_ratio = float(line[1])
            mod_top = np.zeros(mod_nl)
            mod_v = np.zeros(mod_nl)
        if l==13+niter:
            for layer in range(0,mod_nl):
                mod_top[layer] = float(line[layer])
        if l==14+niter:
            for layer in range(0,mod_nl):
                mod_v[layer] = float(line[layer])
        if l==15+niter:
            iclust = int(line[0])
        if l>=16+niter:
            for i in range(ii,ii+8):
                icusp.append(int(line))
                ii = i
        l = l+1

    if len(icusp)>0:
        icusp = np.asarray(icusp)
    inputfile.close()
    ncusp = ii

    # Rearange aiter:
    for i in range(1,niter):
        aiter[i] = aiter[i-1]+aiter[i]

    if not os.path.exists(fn_eve):
        print('File does not exist ' + fn_eve)
    if not os.path.exists(fn_sta):
        raise RuntimeError('File does not exist ' + fn_sta)
    if idata==1 or idata==3:
        if not os.path.exists(fn_cc):
            raise RuntimeError('File does not exist ' + fn_cc)
    if idata==2 or idata==3:
        if not os.path.exists(fn_inp):
            raise RuntimeError('File does not exist ' + fn_ct)  
    
    maxiter = aiter[niter-1]
    # Synthetic Noise:
    noisef_dt=0.

    #Write log output: of newest form
    log.write('INPUT FILES: \ncross dtime data: %s \ncatalog dtime data: %s \nevents: %s \nstations: %s \n\n'
               % (fn_cc,fn_ct,fn_eve,fn_sta))
    log.write('OUTPUT FILES: \ninitial locations: %s \nrelocated events: %s \nevent pair residuals: %s \nstation residuals: %s \nsource parameters: %s \n\n'
               % (fn_loc,fn_reloc,fn_res,fn_stares,fn_srcpar))
    log.write('INPUT PARAMETERS: \nIDATA: %i \nIPHASE: %i \nMAXDIST = %2.4f \nMINOBS_CC: %i \nMINOBS_CT = %i \nISTART: %i \nISOLV: %i \n\n'
               % (idata,iphase,maxdist,minobs_cc,minobs_ct,istart,isolv))

    #aiter[0] = 0  # I'm confused by this so I need to come back
    for i in range(0,niter):
        log.write('ITER: %i - %i \nDAMP: %i \nWT_CCP: %2.4f \nWT_CCS: %2.4f \nMAXR_CC: %2.4f \nMAXD_CC: %2.4f \nWT_CTP: %2.4f \nWT_CTS: %2.4f \nMAXR_CT %2.4f \nMAXD_CT: %2.4f \n\n'
                    % (aiter[i-1]+1,aiter[i],adamp[i],awt_ccp[i],awt_ccs[i],amaxres_cross[i],amaxdcc[i],awt_ctp[i],awt_cts[i],amaxres_net[i],amaxdct[i]))

    # Write crust model
    log.write('MOD_NL: %i \nMOD_RATIO: %2.4f \n' % (mod_nl,mod_ratio))
    log.write('MOD_TOP       MOD_V \n')
    for i in range(0,mod_nl):
        log.write('%4.5f     %4.5f \n' % (mod_top[i],mod_v[i]))

    # Repeat number of clusters/events to relocate
    if iclust==0:
        log.write('Relocate all cluster. \n\n')
        print('Relocate all clusters.')
    else:
        log.write('Relocate cluster number %i \n\n' % iclust)
        print('Relocate cluster number %i' % iclust)
    if ncusp==0:
        log.write('Relocate all events. \n\n')
        print('Relocate all events.')
    else:
        log.write('Relocate %i events \n\n' % ncusp)
        print('Relocate %i events' % ncusp)

    return fn_cc,fn_ct,fn_sta,fn_eve,fn_loc,fn_reloc,fn_res,fn_stares,fn_srcpar, \
    idata,iphase,minobs_cc,minobs_ct,amaxres_cross,amaxres_net,amaxdcc,amaxdct, \
    noisef_dt,maxdist,awt_ccp,awt_ccs,awt_ctp,awt_cts,adamp,istart,maxiter, \
    isolv,niter,aiter,mod_nl,mod_ratio,mod_v,mod_top,iclust,ncusp,icusp


#@profile
def getdata(log,fn_cc,fn_ct,fn_sta,fn_eve,fn_srcpar,
            idata,iphase,ncusp,icusp,
            maxdist,maxsep_ct,maxsep_cc,
            noisef_dt,mod_nl,mod_ratio,mod_v,mod_top):
    """
    This function reads in data input files for hypoDD
    ##########
    INPUTS:
    ##########
    OUTPUTS:

    """
    # Define PI for consistency
    PI = 3.141593
    # Write to log
    datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    log.write('Reading data . . .  %s \n' % datet)
    print('Reading data . . . %s' % datet)

    # Read error event file
    # Bad events are marked in a file called cuspid.err
    if not os.path.exists('cuspid.err'):
        nerr = 0
    else:
        errev = open('cuspid.err','r')
        errevs = errev.readlines()
        nerr = len(errevs)
        cusperr = np.zeros(nerr)
        for ind in range(0,nerr):
            cusperr[ind] = int(errevs[ind])
        errev.close()
    # Read event file
    evfile = open(fn_eve,'r')
    i = 1
    # Begin EQ read loop
    evs = evfile.readlines()
    nev = len(evs)
    ev_date = np.zeros(nev)
    ev_time = np.zeros(nev)
    ev_lat = np.zeros(nev)
    ev_lon = np.zeros(nev)
    ev_dep = np.zeros(nev)
    ev_mag = np.zeros(nev)
    ev_herr = np.zeros(nev)
    ev_zerr = np.zeros(nev)
    ev_res = np.zeros(nev)
    ev_cusp = np.zeros(nev)
    for ind,ev in enumerate(evs):
        ev = ev.split()
        ev = list(filter(None,ev))
        ev_date[ind] = ev[0]
        ev_time[ind] = ev[1]
        ev_lat[ind] = ev[2]
        ev_lon[ind] = ev[3]
        ev_dep[ind] = ev[4]
        ev_mag[ind] = ev[5]
        ev_herr[ind] = ev[6]
        ev_zerr[ind] = ev[7]
        ev_res[ind] = ev[8]
        ev_cusp[ind] = ev[9]
        if ev_date[ind] < 10000000:
            ev_date[ind] = ev_date[ind] + 19000000
        # If EQ is shallower than 10m, for it to 10m
        if ev_dep[ind] < 0.01:
            ev_dep[ind] = 0.01
        # Check if event is on error list
        for i in range(0,nerr):
            if ev_cusp[ind] == cusperr[j]:
                log.write('Note: event in error list: %i \n' % ev_cusp[ind])
                print('Note: event in error list: %i' % ev_cusp[ind])
    # End EQ read loop
    log.write('# events = %i \n' % nev)
    print('# events = %i' % nev)
    if ncusp>0 and ncusp!=nev:
        log.write('>>> Events repeated in selection list or missing/repeated in event file.\n')
        print('>>> Events repeated in selection list or missing/repeated in event file.')
        for i in range(0,ncusp):
            k = 0
            for j in range(0,nev):
                if icusp[i]==ev_cusp[j]:
                    k = k+1
            if k==0:
                log.write('%i is missing \n' % icusp[i])
                print('%i is missing' % icusp[i])
            if k>=2:
                log.write('%i is non-unique \n' % icusp[i])
                raise Exception('Event ID must be unique %i' % icusp[i])
    evfile.close()
    # Get center of event cluster
    clat = np.sum(ev_lat)
    clon = np.sum(ev_lon)
    clat = clat/nev
    clon = clon/nev

    # Get station list
    stafile = open(fn_sta,'r')
    i = 0
    ii = 0
    stas = stafile.readlines()
    nsta = len(stas)
    sta_lab = np.empty(nsta,dtype='object')
    sta_lat = np.zeros(nsta)
    sta_lon = np.zeros(nsta)
    for sta in stas:
        sta = sta.split()
        sta = list(filter(None,sta))
        sta_lab[i] = sta[0]
        sta_lat[i] = mf.atoangle(sta[1])
        sta_lon[i] = mf.atoangle(sta[2])
        # Skip at distances larger than maxdist
        delt,dist,azim = mf.delaz(clat,clon,sta_lat[i],sta_lon[i])
        if dist <= maxdist:
            i = i+1
        ii = i+1
    # Station file now read
    nsta = i
    log.write('# stations total = %i \n' % ii)
    log.write('# stations < maxdist = %i \n' % i)
    print('# stations total = %i' % ii)
    print('# stations < maxdist = %i' % i)
    stafile.close()

    # Check for duplicated stations
    for i in range(0,nsta-1):
        #for j in range(i+1,nsta):
        if sta_lab[i] in sta_lab[i+1:]:
            log.write('%s is listed twice \n' % sta_lab[i])
            raise Exception('%s is listed twice' % sta_lab[i])

    # Declare stuff
    maxdata = nev*nsta
    nccp=0
    nccs=0
    nctp=0
    ncts=0
    dt_sta = np.empty(maxdata,dtype='object')
    dt_dt = np.zeros(maxdata)
    dt_qual = np.zeros(maxdata)
    dt_offs = np.zeros(maxdata)
    dt_c1 = np.zeros(maxdata)
    dt_c2 = np.zeros(maxdata)
    dt_idx = np.zeros(maxdata)
    dt_ista = np.zeros(maxdata)
    dt_ic1 = np.zeros(maxdata)
    dt_ic2 = np.zeros(maxdata)
    
    tmp_ttp = []
    tmp_tts = []
    tmp_xp = []
    tmp_yp = []
    tmp_zp = []
    if idata==0: # Synthetics triggered
        # Generate Synthetics
        elon = np.copy(ev_lon)
        elat = np.copy(ev_lat)

        tmp_ttp,tmp_tts,tmp_xp,tmp_yp,tmp_zp = rt.partials(nev,ev_cusp,elat,elon,ev_dep,nsta,sta_lab,sta_lat,
                                                            sta_lon,mod_nl,mod_ratio,mod_v,mod_top,fn_srcpar)
        # Open Synthetic dtime files
        if iphase==1 or iphase==3:
            synfile = open('dtime.P.syn','w')
            l=1
            for i in range(0,nsta):
                for j in range(0,nev):
                    for k in range(j+1,nev):
                        dt_sta[l] = sta_lab[i]
                        dt_c1[l] = ev_cusp[j]
                        dt_c2[l] = ev_cusp[k]
                        dt_qual[l] = 100
                        dt_idx[l] = 1
                        dt_dt[l] = (tmp_ttp[i,j] - tmp_ttp[i,k])
                        tmp = noisef_dt
                        dtn = 2*tmp*np.random.rand(1) - tmp
                        dt_dt[l] = dt_dt[l] + dtn
                        synfile.write('%s %15.7f %15.7f %4i %4i %9.1f 0 \n' % 
                                      (dt_sta[l],dt_dt[l],-dt_dt[l],dt_c1[l],dt_c2[l],dt_qual[l]))
                        l=l+1
            synfile.close()
            nccp=l-1

        if iphase==2 or iphase==3:
            synfile = open('dtime.S.syn','w')
            l=1
            for i in range(0,nsta):
                for j in range(0,nev):
                    for k in range(j+1,nev):
                        dt_sta[l] = sta_lab[i]
                        dt_c1[l] = ev_cusp[j]
                        dt_c2[l] = ev_cusp[k]
                        dt_qual[l] = 100
                        dt_idx[l] = 1
                        dt_dt[l] = (tmp_tts[i,j] - tmp_tts[i,k])
                        tmp = noisef_dt
                        dtn = 2*tmp*np.random.rand(1) - tmp
                        dt_dt[l] = dt_dt[l] + dtn
                        synfile.write('%s %15.7f %15.7f %4i %4i %9.1f 0 \n' % 
                                      (dt_sta[l],dt_dt[l],-dt_dt[l],dt_c1[l],dt_c2[l],dt_qual[l]))
                        l=l+1
            synfile.close()
            nccs=l-1

            print('# Synthetic P dtimes: %6i' % nccp)
            print('# Synthetic S dtimes: %6i' % nccs)
            log.write('# Synthetic P dtimes: %6i \n' % nccp)
            log.write('# Synthetic S dtimes: %6i \n' % nccs)
    else:
        # Read cross-correlation dtimes
        iicusp = np.argsort(ev_cusp)
        icusp = np.zeros(nev)
        icusp = ev_cusp[iicusp[0:nev]]
        i=0
        iiotc=0
        if idata==1 or idata==3 and len(fn_cc)>1:
            ccfile = open(fn_cc,'r')
            ccs = ccfile.readlines()
            for line in ccs:
                if line[0:1]=='#':
                    line = line.split()
                    line = list(filter(None,line))
                    ic1 = int(line[1])
                    ic2 = int(line[2])
                    otc = float(line[3])
                    iskip = 0
                    # Skip event pairs with no otc
                    if abs(otc+999)<0.01:
                        log.write('No otc for %i %i. Pair Skipped. \n' % (ic1,ic2))
                        print('No otc for %i %i. Pair skipped.' % (ic1,ic2))
                        iiotc=iiotc+1
                        iskip=1
                        continue
                    # Skip event pairs witth events not in event list
                    try:
                        k1 = int(tuple(np.argwhere(icusp==ic1))[0])
                        k2 = int(tuple(np.argwhere(icusp==ic2))[0])
                    except:
                        iskip=1
                        continue
                    # Skip event pairs with large separation distances
                    dlat = ev_lat[iicusp[k1]] - ev_lat[iicusp[k2]]
                    dlon = ev_lon[iicusp[k1]] - ev_lon[iicusp[k2]]
                    offs = np.sqrt((dlat*111.)**2 + 
                                   (dlon*(np.cos(ev_lat[iicusp[k1]]*PI/180.)*111.))**2 +
                                   (ev_dep[iicusp[k1]]-ev_dep[iicusp[k2]])**2)
                    if maxsep_ct>0 and offs>maxsep_ct:
                        iskip=1
                        continue
                else:
                    if iskip==1:
                        continue
                    line = line.split()
                    line = list(filter(None,line))
                    dt_sta[i] = str(line[0])
                    dt_dt[i] = float(line[1])
                    dt_qual[i] = float(line[2])
                    pha = str(line[3])
                    dt_c1[i] = ic1
                    dt_c2[i] = ic2
                    dt_dt[i] = dt_dt[i]-otc

                    # Skip far away data
                    fardata=1
                    if dt_sta[i] in sta_lab:
                    #for j in range(0,nsta):
                        #if dt_sta[i] == sta_lab[j]:
                        fardata = 0
                    if fardata==1:
                        continue

                    # Store time difference
                    if pha=='P':
                        if iphase==2:
                            continue
                        dt_idx[i]=1
                        nccp=nccp+1
                    elif pha=='S':
                        if iphase==1:
                            continue
                        dt_idx[i]=2
                        nccs=nccs+1
                    else:
                        raise Exception('>>> Phase identifier format error')
                    
                    dt_offs[i] = offs

                    i = i+1

            if iphase!=2:
                log.write('# Cross-correlation P dtimes = %7i (no OTC for: %7i) \n' % (nccp,iiotc))
                print('# Cross-correlation P dtimes = %7i (no OTC for: %7i)' % (nccp,iiotc))
            if iphase!=1:
                log.write('# Cross-correlation S dtimes = %7i (no OTC forL %7i) \n' % (nccs,iiotc))
                print('# Cross-correlation S dtimes = %7i (no OTC for: %7i)' % (nccs,iiotc))
            ccfile.close()

        # Read catalog P and S absolute times
        if idata==2 or idata==3 and len(fn_ct)>1:
            ctfile = open(fn_ct,'r')
            cts = ctfile.readlines()
            for line in cts:
                if line[0:1]=='#':
                    line = line.split()
                    line = list(filter(None,line))
                    ic1 = int(line[1])
                    ic2 = int(line[2])
                    iskip = 0
                    # Skip event pairs with events not in event list
                    try:
                        k1 = int(tuple(np.argwhere(icusp==ic1))[0])
                        k2 = int(tuple(np.argwhere(icusp==ic2))[0])
                    except:
                        iskip=1
                        continue
                    # Skip event pairs with large separation distances
                    dlat = ev_lat[iicusp[k1]] - ev_lat[iicusp[k2]]
                    dlon = ev_lon[iicusp[k1]] - ev_lon[iicusp[k2]]
                    offs = np.sqrt((dlat*111.)**2 + 
                                   (dlon*(np.cos(ev_lat[iicusp[k1]])*PI/180)*111.)**2 +
                                   (ev_dep[iicusp[k1]]-ev_dep[iicusp[k2]])**2)
                    if maxsep_ct>0 and offs>maxsep_ct:
                        iskip=1
                        continue
                else:
                    if iskip==1:
                        continue
                    line = line.split()
                    line = list(filter(None,line))
                    dt_sta[i] = str(line[0])
                    dt1 = float(line[1])
                    dt2 = float(line[2])
                    dt_qual[i] = float(line[3])
                    pha = str(line[4])
                    dt_c1[i] = ic1
                    dt_c2[i] = ic2

                    # Store time difference
                    dt_dt[i] = dt1-dt2
                    # Skip far away data
                    fardata=1
                    if dt_sta[i] in sta_lab:
                        fardata=0
                    #for j in range(0,nsta):
                    #    if dt_sta[i] == sta_lab[j]:
                    #        fardata = 0
                    #        break
                    if fardata==1:
                        continue

                    # Store time difference
                    if pha=='P':
                        if iphase==2:
                            continue
                        dt_idx[i]=3
                        nctp=nctp+1
                    elif pha=='S':
                        if iphase==1:
                            continue
                        dt_idx[i]=4
                        ncts=ncts+1
                    else:
                        raise Exception('>>> Phase identifier format error')
                
                    dt_offs[i] = offs
                    i = i+1

            if iphase!=2:
                log.write('# Catalog P dtimes = %7i \n' % (nctp))
                print('# Catalog P dtimes = %7i' % (nctp))
            if iphase!=1:
                log.write('# Catalog S dtimes = %7i \n' % (ncts))
                print('# Catalog S dtimes = %7i' % (ncts))
            ctfile.close()

    ndt = nccp+nccs+nctp+ncts
    dt_sta = dt_sta[0:ndt]
    dt_dt = dt_dt[0:ndt]
    dt_qual = dt_qual[0:ndt]
    dt_offs = dt_offs[0:ndt]
    dt_c1 = dt_c1[0:ndt]
    dt_c2 = dt_c2[0:ndt]
    dt_idx = dt_idx[0:ndt]
    dt_ista = dt_ista[0:ndt]
    dt_ic1 = dt_ic1[0:ndt]
    dt_ic2 = dt_ic2[0:ndt]

    print('# Dtimes total: %8i' % ndt)
    log.write('# Dtimes total: %8i \n' % ndt)
    if ndt==0:
        raise Exception('Dtimes == 0. Stop triggered.')

    # Clean events: dtime match
    #for i in range(0,ndt):
    #    dt_ic1[i] = dt_c1[i] # dt_ic1 and dt_ic2 workspace arrays
    #    dt_ic2[i] = dt_c2[i]
    dt_ic1 = np.sort(dt_c1[0:ndt])
    dt_ic2 = np.sort(dt_c2[0:ndt])
    k=0
    for i in range(0,nev):
        if ev_cusp[i] in dt_ic1 or ev_cusp[i] in dt_ic2:
            ev_date[k] = ev_date[i]
            ev_time[k] = ev_time[i]
            ev_lat[k] = ev_lat[i]
            ev_lon[k] = ev_lon[i]
            ev_dep[k] = ev_dep[i]
            ev_mag[k] = ev_mag[i]
            ev_herr[k] = ev_herr[i]
            ev_zerr[k] = ev_zerr[i]
            ev_res[k] = ev_res[i]
            ev_cusp[k] = ev_cusp[i]
            k = k+1

    nev=k
    print('# Events after dtime match: %10i' % nev)
    log.write('# Events after dtime match: %10i \n' % nev)

    # Clean stations
    sta_itmp = np.zeros(nsta)
    #sta_itmp = np.where(dt_sta==sta_lab,1,0)
    sta_itmp = np.where(np.in1d(sta_lab,dt_sta),1,0)

    #import pdb; pdb.set_trace()
    #or j in range(0,ndt):
    #for i in range(0,nsta):
            #if dt_sta[j]==sta_lab[i]:
    #    if sta_lab[i] in dt_sta:
    #        sta_itmp[i] = 1
    #            break

    k=0
    for i in range(0,nsta):
        if sta_itmp[i]==1:
            sta_lab[k] = sta_lab[i]
            sta_lat[k] = sta_lat[i]
            sta_lon[k] = sta_lon[i]
            k=k+1

    nsta=k
    log.write('# stations = %6i \n' % nsta)
    print('# stations = %6i' % nsta)

    # Indexing station labels and cuspids
    iicusp=np.argsort(ev_cusp[0:nev+1])
    #for i in range(0,nev):
    icusp = ev_cusp[iicusp[0:nev]]
    for i in range(0,ndt):
        #ibreak = 0
        #for j in range(0,nsta):
        try:
            #if dt_sta[i] in sta_lab:
            #import pdb; pdb.set_trace()
            dt_ista[i] = np.where(sta_lab==dt_sta[i])[0][0]
            dt_ic1[i] = iicusp[int(tuple(np.argwhere(icusp==dt_c1[i]))[0])]
            dt_ic2[i] = iicusp[int(tuple(np.argwhere(icusp==dt_c2[i]))[0])]
            #ibreak = 1
        except:
        #else: 
            raise Exception('FATAL ERROR INDEXING. GETDATA.')

    return ev_date,ev_time,ev_cusp,ev_lat,ev_lon,ev_dep, \
    ev_mag,ev_herr,ev_zerr,ev_res,\
    sta_lab,sta_lat,sta_lon,\
    dt_sta,dt_dt,dt_qual,dt_c1,dt_c2,dt_idx,\
    dt_ista,dt_ic1,dt_ic2,dt_offs,\
    nev,nsta,ndt,nccp,nccs,nctp,ncts,\
    tmp_xp,tmp_yp,tmp_zp,tmp_ttp,tmp_tts



