import numpy as np
import os
import sys
from datetime import datetime
import misc_func as mf
import rt_functions as rt
import hypoDD_geodetics as gd
import hypoDD_io as io
import hypoDD_functions as hf
import hypoDD_inversion as inv
import time
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

# Main hypoDD file
#@profile
def hypoDD():
    """
    Taken from hypoDD v1.3 11/2010
    HypoDD v1.3 Author: Felix Waldhauser
    Translated: K. Biegel, katherine.biegel@ucalgary.ca

    Purpose:
    Program to determine high-resolution hypocenter locations using the
    double-difference algorithm. hypoDD incorporates catalog and/or cross
    correlation P- and/or S-wave relative travel-time measurements.
    Residuals between observed and theoretical travel time differences
    (or double-differences = DD) are minimized for pairs
    of earthquakes at each station while linking together all observed
    event/station pairs. A least squares solution (SVD or LSQR) is found
    by iteratively adjusting the vector difference between hypocentral pairs.

    References:
    For a detailed description of the algorithm see:
    Waldhauser, F. and W.L. Ellsworth, A double-difference earthquake
        location algorithm: Method and application to the northern Hayward
        fault, Bull. Seismol. Soc. Am., 90, 1353-1368, 2000.

    For a user guide to hypoDD see USGS open-file report: 
        Waldhauser, F., HypoDD: A computer program to compute double-difference
        earthquake locations,  U.S. Geol. Surv. open-file report , 01-113,
        Menlo Park, California, 2001.
    """

    minwght = 0.00001
    rms_ccold = 0
    rms_ctold = 0
    rms_cc0old = 0
    rms_ct0old = 0
    am_cusp=np.zeros(1000)
    ineg = 0
    rms_cc = 0
    rms_ct = 0
    rms_cc0 = 0
    rms_ct0 = 0

    # Open log file
    log = open('hypoDD.log','w')
    datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    log.write('starting hypoDD (python version) %s \n' % datet)
    print('starting hypoDD (python version) %s' % datet)

    # Get input parameter file name:
    inputs = sys.argv
    if len(inputs)==1:
        print('User Enter Inputs:')
        fn_inp = input('Inputfile location. Default= "hypoDD.inp"')
        if not fn_inp:
            fn_inp = 'hypoDD.inp'
    elif len(inputs)==2:
        fn_inp = inputs[1]
    else:
        raise Error('Input file issues. Run format: run hypoDD_run.py [inputfile]')

    if not os.path.exists(fn_inp):
        raise Error('>> Error opening input parameter file.')

    # Get input parameters:
    [fn_cc,fn_ct,fn_sta,fn_eve,fn_loc,fn_reloc,fn_res,fn_stares,fn_srcpar, \
    idata,iphase,minobs_cc,minobs_ct,amaxres_cross,amaxres_net,amaxdcc,amaxdct, \
    noisef_dt,maxdist,awt_ccp,awt_ccs,awt_ctp,awt_cts,adamp,istart,maxiter, \
    isolv,niter,aiter,mod_nl,mod_ratio,mod_v,mod_top,iclust,ncusp,icusp] = io.getinp(log,fn_inp)

    # Get data
    [ev_date,ev_time,ev_cusp,ev_lat,ev_lon,ev_dep,\
    ev_mag,ev_herr,ev_zerr,ev_res,\
    sta_lab,sta_lat,sta_lon,\
    dt_sta,dt_dt,dt_qual,dt_c1,dt_c2,dt_idx,\
    dt_ista,dt_ic1,dt_ic2,dt_offs,\
    nev,nsta,ndt,nccp,nccs,nctp,ncts,\
    tmp_xp,tmp_yp,tmp_zp,tmp_ttp,tmp_tts] = io.getdata(log,fn_cc,fn_ct,fn_sta,fn_eve,fn_srcpar,
                                                       idata,iphase,ncusp,icusp,
                                                       maxdist,amaxdct[0],amaxdcc[0],
                                                       noisef_dt,mod_nl,mod_ratio,mod_v,mod_top)

    # Clustering
    if (idata==1 and minobs_cc==0) or (idata==2 and minocs_ct==0) or (idata==3 and minobs_ct+minobs_cc==0):
        nclust = 1
        clust = np.zeros((1,nev+1))
        clust[0,0] = nev
        for i in range(0,nev):
            clust[0,i+1] = ev_cusp[i]
        log.write('No clustering performed. \n')
        print('No clustering performed.')
    else:
        [clust,noclust,nclust] = hf.cluster1(log, nev, ndt,idata, minobs_cc, minobs_ct,
                                             dt_c1, dt_c2, ev_cusp)

    # Open files
    loc = open(fn_loc,'w')
    reloc = open(fn_reloc,'w')
    if len(fn_stares)>1:
        stares = open(fn_stares,'w')

    jiter=0 # Counter for iter with no updating (air quakes)
    # Big loop over clusters starts here:
    if iclust!=0:
        if iclust<0 or iclust>nclust:
            raise Exception('Error: invalid cluster number %5i. Must be between 1 and nclust (%5i)' % (iclust,nclust))
        ibeg=0
        iend=iclust
    else:
        ibeg=0
        iend=nclust

    amcusp = np.zeros(1000)

    exav=0
    eyav=0
    ezav=0
    etav=0
    dxav=0
    dyav=0
    dzav=0
    dtav=0
    sta_np = np.zeros(nsta)
    sta_ns = np.zeros(nsta)
    sta_nnp = np.zeros(nsta)
    sta_nns = np.zeros(nsta)
    sta_rmsc = np.zeros(nsta)
    sta_rmsn = np.zeros(nsta)

    for icl in range(ibeg,iend):
        datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log.write('RELOCATION OF CLUSTER: %2i     %s \n' % (iclust,datet))
        print('RELOCATION OF CLUSTER: %2i     %s' % (iclust+1,datet))

        # Get data for each cluster if clustering invoked
        if (nclust!=1) and (minobs_cc>0 or minobs_ct>0):
            ncusp = clust[iclust,0]
            for i in range(0,ncusp):
                icusp[i] = clust[iclust,i+1]

            if idata!=0:
                    [ev_date,ev_time,ev_cusp,ev_lat,ev_lon,ev_dep,\
                     ev_mag,ev_herr,ev_zerr,ev_res,\
                     sta_lab,sta_lat,sta_lon,\
                     dt_sta,dt_dt,dt_qual,dt_c1,dt_c2,dt_idx,\
                     dt_ista,dt_ic1,dt_ic2,dt_offs,\
                     nev,nsta,ndt,nccp,nccs,nctp,ncts,\
                     tmp_xp,tmp_yp,tmp_zp,tmp_ttp,tmp_tts] = io.getdata(log,fn_cc,fn_ct,fn_sta,fn_eve,fn_srcpar,
                                                                        idata,iphase,ncusp,icusp,
                                                                        maxdist,amaxdct[0],amaxdcc[0],
                                                                        noisef_dt,mod_nl,mod_ratio,mod_v,mod_top)

        nccold = nccp+nccs
        nctold = nctp+ncts
        ncc = nccp+nccs
        nct = nctp+ncts
        nevold = nev

        # Get cluster centroid
        sdc0_lat = 0.
        sdc0_lon = 0.
        sdc0_dep = 0.
        sdc0_lat = np.sum(ev_lat)
        sdc0_lon = np.sum(ev_lon)
        sdc0_dep = np.sum(ev_dep)
        sdc0_lat = sdc0_lat/nev
        sdc0_lon = sdc0_lon/nev
        sdc0_dep = sdc0_dep/nev

        log.write('Cluster centroid at: %10.6f  %11.6f  %9.6f \n' % (sdc0_lat,sdc0_lon,sdc0_dep))

        # Set up cartesian coordinates
        gd.setorg(sdc0_lat,sdc0_lon,0.0,0)

        # Get cartesian coordinates for epicenters
        ev_x = np.zeros(nev)
        ev_y = np.zeros(nev)
        ev_z = np.zeros(nev)
        for i in range(0,nev):
            lat = ev_lat[i]
            lon = ev_lon[i]
            [x,y] = gd.sdc2(lat,lon,-1)
            ev_x[i] = x*1000
            ev_y[i] = y*1000
            ev_z[i] = (ev_dep[i]-sdc0_dep)*1000

        log.write('# Events: %5i \n' % nev)

        # Write output (mdat.loc)
        for i in range(0,nev):
            loc.write('%9i %10.6f %11.6f %9.3f %10.1f %10.1f %10.1f %8.1f %8.1f %8.1f %4i %2i %2i %2i %2i %5.2f %4.1f %3i' %
                      (ev_cusp[i],ev_lat[i],ev_lon[i],ev_dep[i],ev_x[i],ev_y[i],ev_z[i],
                       ev_herr[i]*1000,ev_herr[i]*1000,ev_zerr[i]*1000,int(ev_date[i]/10000),
                       int((ev_date[i]//10000)/100),ev_date[i]//100,
                       int(ev_time[i]/1000000),int((ev_time[i]//10000000)/10000),
                       (float(ev_time[i])//10000)/100,ev_mag[i],iclust))

        # Get initial trial sources
        [nsrc,src_cusp,src_lat0,src_lon0,
         src_x0,src_y0,src_z0,src_t0,
         src_lat,src_lon,src_dep,
         src_x,src_y,src_z,src_t] = hf.trialsrc(istart,sdc0_lat,sdc0_lon,sdc0_dep,
                                                nev,ev_cusp,ev_lat,ev_lon,ev_dep)

        print('Initial trial sources = %6i' % nsrc)
        log.write('# Initial trial sources: %6i \n' % nsrc)

        # Loop over iterations starts here:
        # Define each iteration step at which re-weighting starts:
        # this is dynam. since it depends on the number of neg 
        # depths runs before
        for i in range(0,niter):
            aiter[i] = aiter[i] - jiter
        maxiter = maxiter - jiter

        kiter = 0 # counter for iter with data skipping
        jiter = 0 # counter for iter with no updating (air quakes)
        mbad = 0 # counter for air quakes

        iteri = 1
        while iteri<=maxiter:
            datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log.write('=== ITERATION %3i (%3i) %s \n' % (iteri-jiter,iteri,datet))

            # Get weighting parameters for this iteration:
            for i in range(0,niter):
                if iteri<=aiter[i]:
                    break
            maxres_cross = amaxres_cross[i]
            maxres_net = amaxres_net[i]
            maxdcc = amaxdcc[i]
            maxdct = amaxdct[i]
            wt_ccp = awt_ccp[i]
            wt_ccs = awt_ccs[i]
            wt_ctp = awt_ctp[i]
            wt_cts = awt_cts[i]
            damp = adamp[i]

            log.write('Weighting parameters for this iteration: \n')
            log.write('wt_ccp= %7.4f  wt_ccs= %7.4f  maxr_cc= %7.4f  maxd_cc= %7.4f \n' % 
                      (wt_ccp,wt_ccs,maxres_cross,maxdcc))
            log.write('wt_ctp= %7.4f  wt_cts= %7.4f  maxr_ct= %7.4f  maxd_ct= %7.4f  damp= %7.4f \n' % 
                      (wt_ctp,wt_cts,maxres_net,maxdct,damp))

            # Calculate travel times and slowness vectors
            log.write('~ getting partials for %5i stations and %5i sources' % 
                      (nsta,nsrc))

            [tmp_ttp,tmp_tts,
             tmp_xp,tmp_yp,tmp_zp] = rt.partials(nsrc,src_cusp,src_lat,src_lon,src_dep,
                                                 nsta,sta_lab,sta_lat,sta_lon,
                                                 mod_nl,mod_ratio,mod_v,mod_top,fn_srcpar)

            # Get double difference vector:
            [dt_cal,dt_res] = hf.dtres(log,ndt,nsta,nsrc,
                                       dt_dt,dt_idx,
                                       dt_ista,dt_ic1,dt_ic2,
                                       src_cusp,src_t,tmp_ttp,tmp_tts)

            # Get a priori weights and reweight residuals
            [ineg,dt_wt] = hf.weighting(log,ndt,mbad,amcusp,idata,kiter,ineg,
                                   maxres_cross,maxres_net,maxdcc,maxdct,minwght,
                                   wt_ccp,wt_ccs,wt_ctp,wt_cts,
                                   dt_c1,dt_c2,dt_idx,dt_qual,dt_res,dt_offs)

            # Skip outliers and/or air quakes
            if ineg>0:
                [ndt,nev,nsrc,nsta,
                 ev_cusp,ev_date,ev_time,ev_mag,
                 ev_lat,ev_lon,ev_dep,ev_x,ev_y,ev_z,
                 ev_herr,ev_zerr,ev_res,
                 src_cusp, src_lat, src_lon, src_dep,
                 src_lat0, src_lon0,
                 src_x, src_y, src_z, src_t, src_x0, src_y0, src_z0, src_t0,
                 sta_lab,sta_lat,sta_lon,#sta_dist,sta_az,
                 sta_rmsc,sta_rmsn,sta_np,sta_ns,sta_nnp,sta_nns,
                 dt_sta,dt_c1,dt_c2,dt_idx,dt_dt,dt_qual,dt_cal,
                 dt_ista,dt_ic1,dt_ic2,
                 dt_res,dt_wt,dt_offs,
                 tmp_ttp,tmp_tts,tmp_xp,tmp_yp,tmp_zp,nct,ncc] = hf.skip(log,kiter,minwght,
                                                                         ndt,nev,nsrc,nsta,
                                                                         ev_cusp,ev_date,ev_time,ev_mag,
                                                                         ev_lat,ev_lon,ev_dep,ev_x,ev_y,ev_z,
                                                                         ev_herr,ev_zerr,ev_res,
                                                                         src_cusp, src_lat, src_lon, src_dep,
                                                                         src_lat0, src_lon0,
                                                                         src_x, src_y, src_z, src_t, src_x0, src_y0, src_z0, src_t0,
                                                                         sta_lab,sta_lat,sta_lon,#sta_dist,sta_az,
                                                                         sta_rmsc,sta_rmsn,sta_np,sta_ns,sta_nnp,sta_nns,
                                                                         dt_sta,dt_c1,dt_c2,dt_idx,dt_dt,dt_qual,dt_cal,
                                                                         dt_ista,dt_ic1,dt_ic2,
                                                                         dt_res,dt_wt,dt_offs,
                                                                         tmp_ttp,tmp_tts,tmp_xp,tmp_yp,tmp_zp,nct,ncc)
                # Skip cluster if less than 2 events
                if nev<2:
                    log.write('Cluster has less than 2 events. \n')
                    print('Cluster has less than 2 events.')
                    break
            else:
                log.write('No data skipped. \n')

            # Get initial residual statistics (avrg,rms,var...)
            if iteri==1:
                resvar1=-999
                [rms_cc,rms_ct,rms_cc0,rms_ct0, rms_ccold,rms_ctold,
                 rms_cc0old,rms_ct0old,resvar1] = hf.resstat(log,idata,ndt,nev,dt_res,dt_wt,dt_idx,
                                                             rms_cc,rms_ct,rms_cc0,rms_ct0,
                                                             rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,
                                                             resvar1)

            # Inversion
            if isolv==1:
                [src_cusp,src_dx,src_dy,src_dz,src_dt,
                 src_ex,src_ey,src_ez,src_et,
                 exav,eyav,ezav,etav,
                 dxav,dyav,dzav,dtav,
                 rms_cc,rms_ct,rms_cc0,rms_ct0,
                 rms_ccold,rms_ctold,rms_cc0old,rms_ct0old] = inv.fast_svd(log,iteri,ndt,nev,nsrc,damp,mod_ratio,
                                                                           idata,ev_cusp,src_cusp,dt_res,dt_wt,
                                                                           dt_ista,dt_ic1,dt_ic2,
                                                                           exav, eyav, ezav, etav, dxav, dyav, dzav, dtav,
                                                                           rms_cc,rms_ct,rms_cc0,rms_ct0,
                                                                           rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,
                                                                           tmp_xp,tmp_yp,tmp_zp,dt_idx)
            if isolv==2:
                [src_cusp,src_dx,src_dy,src_dz,src_dt, 
                 src_ex,src_ey,src_ez,src_et,
                 exav,eyav,ezav,etav,dxav,dyav,dzav,dtav,
                 rms_cc,rms_ct,rms_cc0,rms_ct0,
                 rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,
                 acond] = inv.lsfit_lsqr(log,iteri,ndt,nev,nsrc,damp,mod_ratio, 
                                         idata,ev_cusp,src_cusp,dt_res,dt_wt,
                                         dt_ista,dt_ic1,dt_ic2,
                                         exav,eyav,ezav,etav,dxav,dyav,dzav,dtav,
                                         rms_cc,rms_ct,rms_cc0,rms_ct0,
                                         rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,
                                         tmp_xp,tmp_yp,tmp_zp,dt_idx)


            # Check for air quakes
            mbad = 0
            k = 1
            for i in range(0,nsrc):
                if (src_dep[i]+src_dz[i]/1000)<0:
                    log.write('>>> Warning: negative depth - %12i \n' % ev_cusp[i])
                    amcusp[k] = ev_cusp[i]
                    k = k+1
                    if k>1000:
                        raise Exception('>>> More than 1000 air quakes.')
            mbad = k-1 # Number of neg depth events

            # Update iteration numbers:
            iskip=0
            if mbad>0:
                for i in range(0,niter):
                    aiter[i] = aiter[i] + 1
                jiter = jiter+1 # iteration with no update
                maxiter = maxiter+1

                log.write('Number of air quakes (AQ) = %i \n' % mbad)
                if (nsrc-mbad)<=1:
                    log.write('Warning: number of non-airquakes < 2.  Skipping cluster. \n')
                    print('Warning: number of non-airquakes < 2.  Skipping cluster. \n')
                    continue
                iskip=1

            # Update source parameterds:
            if iskip==0:
                xav = 0 # mean centroid shift
                yav = 0
                zav = 0
                tav = 0
                alon = 0
                alat = 0
                adep = 0
                if nsrc==1:
                    nsrc=nev
                for i in range(0,nsrc):
                    src_cusp[i] = ev_cusp[i]
                    # Update absolute source parameters
                    src_x[i] = src_x[i] + src_dx[i]
                    src_y[i] = src_y[i] + src_dy[i]
                    src_z[i] = src_z[i] + src_dz[i]
                    src_t[i] = src_t[i] + src_dt[i]
                    # Update absolute source locations
                    src_dep[i] = src_dep[i] + src_dz[i]/1000
                    [lat,lon] = gd.sdc2(src_x[i]/1000,src_y[i]/1000,1)
                    src_lon[i] = lon
                    src_lat[i] = lat
                    alon = lon+alon
                    alat = lat+alat
                    adep = adep+src_dep[i]
                    # Mean centroid shift
                    xav = xav + (src_x[i] - src_x0[i])
                    yav = yav + (src_y[i] - src_y0[i])
                    zav = zav + (src_z[i] - src_z0[i])
                    tav = tav + (src_t[i] - src_t0[i])
                xav = xav/nsrc
                yav = yav/nsrc
                zav = zav/nsrc
                tav = tav/nsrc
                alon = alon/nsrc
                alat = alat/nsrc
                adep = adep/nsrc

                log.write('Cluster centroid at: %10.6f  %11.6f  %9.6f \n' % (alat,alon,adep))
                log.write('Mean centroid (origin) shift in x,y,z,t [m,ms]: %7.1f,%7.1f,%7.1f,%7.1f \n' % (xav,yav,zav,tav))

                # Get interevent distanve for each observation and average signal coherency
                cohav = 0
                picav = 0
                j = nct
                k = ncc
                ncc = 0
                nct = 0
                for i in range(0,ndt):
                    dt_offs[i] = np.sqrt((src_x[int(dt_ic1[i])]-src_x[int(dt_ic2[i])])**2+
                                         (src_y[int(dt_ic1[i])]-src_y[int(dt_ic2[i])])**2+
                                         (src_z[int(dt_ic1[i])]-src_z[int(dt_ic2[i])])**2)
                    if dt_idx[i]<=2:
                        cohav = cohav + np.sqrt(dt_qual[i])
                        ncc = ncc+1
                    else:
                        picav = picav + dt_qual[i]
                        nct = nct+1

                cohav = cohav/ncc
                picav = picav/nct
                log.write('More: \n')
                log.write(' mean phase coherency = %5.3f \n' % cohav)
                log.write(' mean pick quality = %5.3f \n' % picav)

                # Get number of observations and mean residual at each station
                tmpr1 = 0
                tmpr2 = 0
                sta_np = np.zeros(nsta,dtype='int')
                sta_ns = np.zeros(nsta,dtype='int')
                sta_nnp = np.zeros(nsta,dtype='int')
                sta_nns = np.zeros(nsta,dtype='int')
                sta_rmsc = np.zeros(nsta)
                sta_rmsn = np.zeros(nsta)
                for i in range(0,nsta):
                    for j in range(0,ndt):
                        if i==dt_ista[j]:
                            sta_rmsc[i] = np.sum(np.where(dt_idx<=2,dt_res[j]**2,0))
                            sta_rmsn[i] = np.sum(np.where(dt_idx>2,dt_res[j]**2,0))
                            sta_np[i] = (dt_idx==1).sum()
                            sta_ns[i] = (dt_idx==2).sum()
                            sta_nnp[i] = (dt_idx==3).sum()
                            sta_nns[i] = (dt_idx==4).sum()

                    if (sta_np[i]+sta_ns[i])>0:
                        sta_rmsc[i] = np.sqrt(sta_rmsc[i]/(sta_np[i]+sta_ns[i]))
                    if (sta_nnp[i]+sta_nns[i])>0:
                        sta_rmsn[i] = np.sqrt(sta_rmsn[i]/(sta_nnp[i]+sta_nns[i]))
                    if sta_rmsc[i]>tmpr1:
                        tmpr1 = sta_rmsc[i]
                        k = i
                    if sta_rmsn[i]>tmpr2:
                        tmpr2 = sta_rmsn[i]
                        l = i

                tmpr1 = tmpr1*1000
                tmpr2 = tmpr2*1000
                if idata==1 or idata==3:
                    log.write(' station with largest cc rms: %s = %7.0f ms (RMSST) \n' % (sta_lab[k],tmpr1))
                if idata==2 or idata==3:
                    log.write(' station with largest ct rms: %s = %7.0f ms (RMMST) \n' % (sta_lab[l],tmpr2))

                # write output scratch mdat.reloc
                i = iteri-jiter
                str80 = '%s.%03i.%03i' % (fn_reloc,iclust,i)
                relocs = open(str80,'w')
                for i in range(0,nev):
                    relocs.write('%9i %10.6f %11.6f %9.3f %10.1f %10.1f %10.1f %8.1f %8.1f %8.1f %4i %2i %2i %2i %2i %6.3f %4.1f %3i \n' %
                                 (src_cusp[i],src_lat[i],src_lon[i],src_dep[i],src_x[i],src_y[i],src_z[i],src_ex[i],src_ey[i],src_ez[i],
                                  int(ev_date[i]/10000),int(np.mod(ev_date[i],1000000)/100),np.mod(ev_date[i],100),int(ev_time[i]/1000000),
                                  int(np.mod(ev_time[i],1000000)/10000),np.mod(float(ev_time[i]),10000.)/100,ev_mag[i],iclust))
                relocs.close()
                log.write('Relocation results for this iteration are stored in %s \n\n\n' % str80)

            # Standard output
            if mbad>0:
                str3 = '   '
            else:
                n = iteri-jiter
                if n<1000:
                    str3=str(n)
                if n<100:
                    str3 = ' %2i' % n
                if n<10:
                    str3 = '  %1i' % n
            if isolv==1 and idata==3:
                if iteri==1:
                    print(' IT    EV   CT   CC      RMSCT        RMSCC   RMSST   DX   DY   DZ   DT   OS   AQ')
                    print('        %    %    %    ms      %    ms      %    ms    m    m    m   ms    m')
                print('%2i%s %3i  %3i  %3i %5i  %5s %5i  %5s %5i %4i %4i %4i %4i %4i %4i' %
                      (iteri,str3,np.rint(nev*100./nevold),np.rint(nct*100./nctold),
                       np.rint(ncc*100./nccold),
                       np.rint(rms_ct*1000),str(round((rms_ct-rms_ctold)*100./rms_ctold,1)).rjust(5),
                       np.rint(rms_cc*1000.),str(round((rms_cc-rms_ccold)*100./rms_ccold,1)).rjust(5),
                       np.rint(np.maximum(tmpr1,tmpr2)),np.rint(dxav),np.rint(dyav),
                       np.rint(dzav),np.rint(dtav),
                       np.rint(np.maximum(np.abs(xav),np.maximum(np.abs(yav),np.abs(zav)))),mbad))
            if isolv==1 and idata==2:
                if iteri==1:
                    print(' IT    EV   CT      RMSCT   RMSST   DX   DY   DZ   DT   OS   AQ')
                    print('        %    %    ms      %    ms    m    m    m   ms    m')
                print('%2i%s %3i  %3i %5i  %5s %5i %4i %4i %4i %4i %4i %4i' %
                      (iteri,str3,np.rint(nev*100./nevold),np.rint(nct*100./nctold),
                       np.rint(rms_ct*1000),str(round((rms_ct-rms_ctold)*100./rms_ctold,1)).rjust(5),
                       np.rint(np.maximum(tmpr1,tmpr2)),np.rint(dxav),np.rint(dyav),
                       np.rint(dzav),np.rint(dtav),
                       np.rint(np.maximum(np.abs(xav),np.maximum(np.abs(yav),np.abs(zav)))),mbad))
            if isolv==1 and idata==1:
                if iteri==1:
                    print(' IT    EV   CC      RMSCC   RMSST   DX   DY   DZ   DT   OS   AQ')
                    print('        %    %    ms      %    ms    m    m    m   ms    m')
                print('%2i%s %3i  %3i  %3i %5i  %5s %5i %4i %4i %4i %4i %4i %4i' %
                      (iteri,str3,np.rint(nev*100./nevold),
                       np.rint(ncc*100./nccold),
                       np.rint(rms_cc*1000.),str(round((rms_cc-rms_ccold)*100./rms_ccold,1)).rjust(5),
                       np.rint(np.maximum(tmpr1,tmpr2)),np.rint(dxav),np.rint(dyav),
                       np.rint(dzav),np.rint(dtav),
                       np.rint(np.maximum(np.abs(xav),np.maximum(np.abs(yav),np.abs(zav)))),mbad))
            if isolv==2 and idata==3:
                if iteri==1:
                    print(' IT    EV   CT   CC      RMSCT        RMSCC   RMSST   DX   DY   DZ   DT   OS   AQ   CND')
                    print('        %    %    %    ms      %    ms      %    ms    m    m    m   ms    m')
                print('%2i%s %3i  %3i  %3i %5i  %5s %5i  %5s %5i %4i %4i %4i %4i %4i %4i %5i' %
                      (iteri,str3,np.rint(nev*100./nevold),np.rint(nct*100./nctold),
                       np.rint(ncc*100./nccold),
                       np.rint(rms_ct*1000),str(round((rms_ct-rms_ctold)*100./rms_ctold,1)).rjust(5),
                       np.rint(rms_cc*1000.),str(round((rms_cc-rms_ccold)*100./rms_ccold,1)).rjust(5),
                       np.rint(np.maximum(tmpr1,tmpr2)),np.rint(dxav),np.rint(dyav),
                       np.rint(dzav),np.rint(dtav),
                       np.rint(np.maximum(np.abs(xav),np.maximum(np.abs(yav),np.abs(zav)))),mbad,np.rint(acond)))
            if isolv==2 and idata==2:
                if iteri==1:
                    print(' IT    EV   CT      RMSCT   RMSST   DX   DY   DZ   DT   OS   AQ   CND')
                    print('        %    %    ms      %    ms    m    m    m   ms    m')
                print('%2i%s %3i  %3i %5i  %5s %5i %4i %4i %4i %4i %4i %4i %5i' %
                      (iteri,str3,np.rint(nev*100./nevold),np.rint(nct*100./nctold),
                       np.rint(rms_ct*1000),str(round((rms_ct-rms_ctold)*100./rms_ctold,1)).rjust(5),
                       np.rint(np.maximum(tmpr1,tmpr2)),np.rint(dxav),np.rint(dyav),
                       np.rint(dzav),np.rint(dtav),
                       np.rint(np.maximum(np.abs(xav),np.maximum(np.abs(yav),np.abs(zav)))),mbad,np.rint(acond)))
            if isolv==2 and idata==1:
                if iteri==1:
                    print(' IT    EV   CC      RMSCC   RMSST   DX   DY   DZ   DT   OS   AQ   CND')
                    print('        %    %    ms      %    ms    m    m    m   ms    m')
                print('%2i%s %3i  %3i %5i  %5s %5i %4i %4i %4i %4i %4i %4i %5i' %
                      (iteri,str3,np.rint(nev*100./nevold),
                       np.rint(ncc*100./nccold),
                       np.rint(rms_cc*1000.),str(round((rms_cc-rms_ccold)*100./rms_ccold,1)).rjust(5),
                       np.rint(np.maximum(tmpr1,tmpr2)),np.rint(dxav),np.rint(dyav),
                       np.rint(dzav),np.rint(dtav),
                       np.rint(np.maximum(np.abs(xav),np.maximum(np.abs(yav),np.abs(zav)))),mbad,np.rint(acond)))

            datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log.write('Iteration %2i finished %s \n' % (iteri,datet))
            if iteri<=maxiter:
                iteri = iteri+1


        # Update origin time
        print('Writing out results')
        src_t[i] = src_t[i]/1000. # change to s
        if src_t[i]>5:
            print('Warning: org time diff > 5s for %i' % src_cusp[i])
        iyr = int(ev_date[i]/10000)
        imo = int(np.mod(ev_date[i],10000)/100)
        idy = int(np.mod(ev_date[i],100))
        ihr = int(ev_time[i]/1000000)
        imn = int(np.mod(ev_time[i],1000000)/10000)
        itf = hf.juliam(iyr,imo,idy,ihr,imn)

        sc = np.mod(float(ev_time[i]),10000.)/100 - src_t[i]
        itf = itf + int(sc/20.)
        sc = sc - int(sc/60.)*60.
        if sc<0:
            itf = itf-1
            sc = 60. + sc
        [iyr,imo,idy,ihr,imn] = hf.datum(itf,iyr,imo,idy,ihr,imn)
        ev_date[i] = iyr*10000 + imo*100 + idy
        ev_time[i] = ihr*1000000 + imn*10000 + int(sc*100)

        # get # of obs per event:
        src_np = np.zeros(nev)
        src_ns = np.zeros(nev)
        src_nnp = np.zeros(nev)
        src_nns = np.zeros(nev)
        src_rmsc = np.zeros(nev)
        src_rmsn = np.zeros(nev)
        for i in range(0,nev):
            src_np[i] = 0
            src_ns[i] = 0
            src_nnp[i] = 0
            src_nns[i] = 0
            src_rmsc[i] = 0
            src_rmsn[i] = 0
        temp_cc = open('temp_cc.txt','w')
        temp_ct = open('temp_ct.txt','w')
        for i in range(0,ndt):
            if dt_idx[i]==1:
                src_np[int(dt_ic1[i])] = src_np[int(dt_ic1[i])]+1
                src_np[int(dt_ic2[i])] = src_np[int(dt_ic2[i])]+1
            if dt_idx[i]==2:
                src_ns[int(dt_ic1[i])] = src_ns[int(dt_ic1[i])]+1
                src_ns[int(dt_ic2[i])] = src_ns[int(dt_ic2[i])]+1
            if dt_idx[i]<=2:
                src_rmsc[int(dt_ic1[i])] = src_rmsc[int(dt_ic1[i])] + dt_res[i]**2
                src_rmsc[int(dt_ic2[i])] = src_rmsc[int(dt_ic2[i])] + dt_res[i]**2
                temp_cc.write('%i %i %f \n' % (dt_ic1[i],dt_ic2[i],dt_res[i]**2))
            if dt_idx[i]==3:
                src_nnp[int(dt_ic1[i])] = src_nnp[int(dt_ic1[i])]+1
                src_nnp[int(dt_ic2[i])] = src_nnp[int(dt_ic2[i])]+1
            if dt_idx[i]==4:
                src_nns[int(dt_ic1[i])] = src_nns[int(dt_ic1[i])]+1
                src_nns[int(dt_ic2[i])] = src_nns[int(dt_ic2[i])]+1
            if dt_idx[i]>=3:
                src_rmsn[int(dt_ic1[i])] = src_rmsn[int(dt_ic1[i])] + dt_res[i]**2
                src_rmsn[int(dt_ic2[i])] = src_rmsn[int(dt_ic2[i])] + dt_res[i]**2
                temp_ct.write('%i %i %f \n' % (dt_ic1[i],dt_ic2[i],dt_res[i]**2))
        for i in range(0,nev):
            if (src_np[i]+src_ns[i])>0:
                src_rmsc[i] = np.sqrt(src_rmsc[i]/(src_np[i]+src_ns[i]))
            else:
                src_rmsc[i] = -9
            if (src_nnp[i]+src_nns[i])>0:
                src_rmsn[i] = np.sqrt(src_rmsn[i]/(src_nnp[i]+src_nns[i]))
            else:
                src_rmsn[i] = -9

        # output final residuals: mdat.res
        res = open(fn_res,'w')
        res.write('STA           DT        C1        C2    IDX     QUAL    RES [ms]   WT         OFFS \n')
        for j in range(0,ndt):
            res.write('%7s %12.7f %9i %9i %1i %9.4f %12.6f %11.6f %8.1f \n' %
                      (dt_sta[j],dt_dt[j],dt_c1[j],dt_c2[j],dt_idx[j],dt_qual[j],
                       dt_res[j]*1000,dt_wt[j],dt_offs[j]))
        res.close()

        # output final locations: mdat.reloc
        relocs_all = open(fn_reloc,'w')
        for i in range(0,nev):
            relocs_all.write('%9i %10.6f %10.6f %9.3f %10.1f %10.1f %10.1f %8.1f %8.1f %8.1f %4i %2i %2i %2i %2i %6.3f %4.1f %5i %5i %5i %5i %6.3f %6.3f %3i \n' %
                             (src_cusp[i],src_lat[i],src_lon[i],src_dep[i],src_x[i],src_y[i],src_z[i],
                              src_ex[i],src_ey[i],src_ez[i],np.rint(ev_date[i]/10000),
                              np.rint(np.mod(ev_date[i],10000)/100),np.mod(ev_date[i],100),
                              np.rint(ev_time[i]/1000000),np.rint(np.mod(ev_time[i],1000000)/10000),
                              np.mod(float(ev_time[i]),10000.)/100.,ev_mag[i],
                              src_np[i],src_ns[i],src_nnp[i],src_nns[i],src_rmsc[i],src_rmsn[i],iclust))

        # ouput stations: mdat.station

        if len(fn_stares)>1:
            for i in range(0,nsta):
                stares.write('%7s %9.4f %9.4f %7i %7i %7i %7i %9.4f %9.4f %3i \n' %
                             (sta_lab[i],sta_lat[i],sta_lon[i],#sta_dist[i],sta_az[i],
                             sta_np[i],sta_ns[i],sta_nnp[i],sta_nns[i],
                             sta_rmsc[i],sta_rmsn[i],iclust))
        log.close()
        if len(fn_stares)>1:
            stares.close()
        relocs_all.close()

start = time.time()
hypoDD()
print('Time: ',time.time()-start)