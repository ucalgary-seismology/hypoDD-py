import numpy as np
from datetime import datetime
import hypoDD_geodetics as gd
import line_profiler
import atexit
from numba import jit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

### Assorted functions from hypoDD v1.3

#@profile
def cluster1(log, nev, ndt, idata, minobs_cc, minobs_ct,
             dt_c1, dt_c2, ev_cusp):
    """
    Cluster events.
    """
    log.write('Clustering....\n')
    print('Clustering...')

    # Set up event-pair arrays
    apair_n = np.empty(int((nev*(nev-1))/2),dtype='int8')
    k=0
    for i in range(1,nev): # Lower triangle of matrix
        for j in range(0,i-1):
            apair_n[k] = 0
            k = k+1

    #icusp = np.zeros(nev,dtype='int8')
    icusp = np.copy(ev_cusp[0:nev])
    icusp = np.sort(icusp)
                  
    for i in range(0,ndt):
        j = int(tuple(np.argwhere(icusp==dt_c1[i]))[0])
        k = int(tuple(np.argwhere(icusp==dt_c2[i]))[0])
        if k>j:
            # Map into lower triangle
            kk = k
            k = j
            j = kk
        apair_n[int(((j)**2-(j))/2)+k] = apair_n[int(((j)**2-(j))/2)+k] + 1
    if idata==0 or idata==1:
        minobs_ct=0
    if idata==0 or idata==2:
        minobs_cc=0

    # Initialize array acl to store cluster index for each event
    acl = np.zeros(nev)
    for i in range(0,nev):
        acl[i] = nev+1
    k=0
    n=0
    for i in range(1,nev):
        for j in range(0,i-1):
            if apair_n[k]>=(minobs_cc+minobs_ct):
                if acl[i]<acl[j]:
                    if acl[j]==(nev+1):
                        acl[j] = acl[i]
                    else:
                        ia = acl[j]
                        for ii in range(0,i):
                            if acl[ii]==ia:
                                acl[ii]=acl[i]
                elif acl[j]<acl[i]:
                    if acl[i]==(nev+1):
                        acl[i] = acl[j]
                    else:
                        ia=acl[i]
                        for ii in range(0,i):
                            if acl[ii]==ia:
                                acl[ii]=acl[j]
                elif acl[i]==(nev+1):
                    n=n+1
                    acl[i] = n
                    acl[j] = n
            k = k+1

    # Store event keys in cluster matrix clust[]
    acli = np.argsort(acl)
    clust = np.zeros((nev,nev+1))
    noclust = np.zeros(nev)
    n = 1
    nn = 2
    iskip=0
    if acl[acli[0]]==(nev+1):
        # No events clustered
        i=1
        iskip=1
    else:
        clust[n-1,nn-1] = icusp[acli[0]]
    if iskip==0:
        for i in range(1,nev):
            if acl[acli[i]] > acl[acli[i-1]]:
                clust[n-1,0] = nn-1
                n = n+1
                nn = 1
            if acl[acli[i]]==(nev+1):
                iskip==1
                break
            nn = nn+1
            clust[n-1,nn-1] = icusp[acli[i]]
    if iskip==0:
        clust[n-1,0] = nn-1
        nclust = n
        noclust[0] = 0
    elif iskip==1:
        nclust = n-1
        for j in range(i,nev):
            noclust[j-i+2] = icusp[acli[j]]
        noclust[0] = nev-i+1
    else:
        raise Exception('Clustering problem.')
    # Sort-biggest cluster first
    if nclust>1:
        for i in range(0,nclust-1):
            for j in range(i+1,nclust):
                if clust[i,0] <= clust[j,0]:
                    for k in range(0,clust[i,0]+1):
                        clust[nev,k] = clust[i,k]
                    for k in range(0,clust[j,0]+1):
                        clust[i,k] = clust[j,k]
                    for k in range(0,clust[nev,0]+1):
                        clust[j,k] = clust[nev,k]
    k = 0
    for i in range(0,nclust):
        k = k+clust[i,0]

    print('Clustered events: %5i' % k)
    print('Isolated events: %5i' % noclust[0])
    print('# clusters: %5i' % nclust)
    k = 0
    for i in range(0,nclust):
        print('Cluster %4i: %5i events' % (i+1,clust[i,0]))
        k = k+clust[i,0]

    log.write('# Clustered events: %5i \n' % k)
    log.write('# Isolated events: %5i \n' % noclust[0])
    for i in range(1,int(noclust[0]+1)):
        log.write(' %15s ' % noclust[i])
    log.write('# Clusters= %5i, for min. number of links set to %5i \n' % (nclust,minobs_ct+minobs_cc))
    k = 0
    for i in range(0,nclust):
        log.write('Cluster %5i: %5i events \n' % (i,clust[i,0]))
        for j in range(1,int(clust[i,0])+1):
            log.write('%15s ' % clust[i,j])
        k = k+clust[i,0]
    log.write('\n\n')

    if nclust==0:
        raise Exception('No clusters.')

    return clust,noclust,nclust


#@jit
def trialsrc(istart,sdc0_lat,sdc0_lon,sdc0_dep,
             nev,ev_cusp,ev_lat,ev_lon,ev_dep):
    """
    Set up source locations if istart=1 or for 
    synthetic models
    """
    # Set up parameters for initial inversion
    if istart==1:
        # Cluster center as initial trial source
        nsrc = 1
        src_cusp = np.copy(ev_cusp)
        src_lon = np.full((nev),sdc0_lon)
        src_lat = np.full((nev),sdc0_lat)
        src_dep = np.full((nev),sdc0_dep)
        src_x = np.zeros(nev)
        src_y = np.zeros(nev)
        src_z = np.zeros(nev)
        src_t = np.zeros(nev)
        src_lon0 = np.full((nev),sdc0_lon)
        src_lat0 = np.full((nev),sdc0_lat)
        src_x0 = np.zeros(nev)
        src_y0 = np.zeros(nev)
        src_z0 = np.zeros(nev)
        src_t0 = np.zeros(nev)
    else:
        # Catalog sources as initial trial source
        # Add noise for synthetic data mode
        src_x = np.zeros(nev)
        src_y = np.zeros(nev)

        nsrc=nev
        src_cusp = np.copy(ev_cusp)
        src_lon = np.copy(ev_lon)
        src_lat = np.copy(ev_lat)
        src_dep = np.copy(ev_dep)
        for i in range(0,nev):
            [x,y] = gd.sdc2(src_lat[i],src_lon[i],-1)
            src_x[i] = x*1000.
            src_y[i] = y*1000.
        src_z = np.full((nev),(ev_dep-sdc0_dep)*1000.)
        src_t = np.zeros(nev)
        src_lon0 = np.copy(ev_lon)
        src_lat0 = np.copy(ev_lat)
        src_x0 = np.copy(src_x)
        src_y0 = np.copy(src_y)
        src_z0 = np.copy(src_z)
        src_t0 = np.copy(src_t)
            
    return [nsrc,src_cusp,src_lat0,src_lon0,
            src_x0,src_y0,src_z0,src_t0,
            src_lat,src_lon,src_dep,
            src_x,src_y,src_z,src_t]


#@profile
#@jit
def dtres(log, ndt, stdim, nsrc, dt_dt, dt_idx,
          dt_ista, dt_ic1, dt_ic2, src_cusp, src_t,
          tmp_ttp, tmp_tts):
    """
    Calculates difference vector dt_cal and double
    difference vector dt_dt
    """
    dt_res = np.zeros(ndt)
    dt_cal = np.zeros(ndt)
    dt_ista = dt_ista.astype('int')
    dt_ic1 = dt_ic1.astype('int')
    dt_ic2 = dt_ic2.astype('int')

    if nsrc==1:
        # Single source
        dt_res = np.copy(dt_dt)
    else:
        # Multiple sources
        tt1 = 0.
        tt2 = 0.
        for i in range(0,ndt):
            if dt_idx[i]==1 or dt_idx[i]==3:
                # P-phase
                tt1 = tmp_ttp[dt_ista[i],dt_ic1[i]] - src_t[dt_ic1[i]]/1000
                tt2 = tmp_ttp[dt_ista[i],dt_ic2[i]] - src_t[dt_ic2[i]]/1000
            elif dt_idx[i]==2 or dt_idx[i]==4:
                # S phase
                tt1 = tmp_tts[dt_ista[i],dt_ic1[i]] - src_t[dt_ic1[i]]/1000
                tt2 = tmp_tts[dt_ista[i],dt_ic2[i]] - src_t[dt_ic2[i]]/1000
            if tt1==0. or tt2==0.:
                raise Exception('Fatal Error (theor tt)')

            dt_cal[i] = tt1-tt2
            dt_res[i] = dt_dt[i] - dt_cal[i]

    return dt_cal,dt_res


#@profile
def weighting(log, ndt, mbad, amcusp, idata, kiter, ineg,
              maxres_cross, maxres_net, maxdcc, maxdct, minwght,
              wt_ccp, wt_ccs, wt_ctp, wt_cts,
              dt_c1, dt_c2, dt_idx, dt_qual, dt_res, dt_offs):
    """
    Determines a priori weights and re-weights them
    """
    # Synthetics
    if idata==0:
        dt_wt = np.ones(ndt)
    else:
        dt_wt = np.zeros(ndt)

    # ---- Get a priori data weights
    # all the quality transfer is done in getdata
    ineg = 0 # flag if neg weights exist
    for i in range(0,ndt):
        if dt_idx[i]==1:
            dt_wt[i] = wt_ccp*dt_qual[i]
        elif dt_idx[i]==2:
            dt_wt[i] = wt_ccs*dt_qual[i]
        elif dt_idx[i]==3:
            dt_wt[i] = wt_ctp*dt_qual[i]
        elif dt_idx[i]==4:
            dt_wt[i] = wt_cts*dt_qual[i]

        for j in range(0,mbad):
            if dt_c1[i]==amcusp[j] or dt_c2[i]==amcusp[j]:
                dt_wt[i]=0.
                ineg=1

    # Reweighting
    datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    dt_tmp = np.zeros(ndt)
    if ((idata==1 or idata==3) and (maxres_cross!=-9 or maxdcc!=-9)) or ((idata==2 or idata==3) and (maxres_net!=-9 or maxdct!=-9)):
        log.write('Reweighting ... %s \n' % datet)

        # Get median and MAD of residuals
        if idata==3:
            if maxres_cross>=1:
                # Cross data
                k=0
                for i in range(0,ndt):
                    if dt_idx[i]<=2:
                        dt_tmp[k] = dt_res[i]
                        k = k+1
                med_cc = np.median(dt_tmp[0:k])
                dt_tmp = np.abs(dt_tmp - med_cc)
                mad_cc = np.median(dt_tmp[0:k])
                mad_cc = mad_cc/0.67449 # MAD for gaussian
            if maxres_net >= 1:
                k=0
                for i in range(0,ndt):
                    if dt_idx[i]>=3:
                        dt_tmp[k] = dt_res[i]
                        k = k+1
                med_ct = np.median(dt_tmp[0:k])
                dt_tmp = np.abs(dt_tmp - med_ct)
                mad_ct = np.median(dt_tmp[0:k])
                mad_ct = mad_ct/0.67449 # MAD for gaussian
        elif (idata==1 and maxres_cross>=1) or (idata==2 and maxres_net>=1):
            dt_tmp = np.copy(dt_res)
            med_cc = np.median(dt_tmp[0:ndt+1])
            dt_tmp = np.abs(dt_tmp-med_cc)
            mad_cc = np.median(dt_tmp[0:ndt+1])
            mad_cc = mad_cc/0.67449
            if idata==2:
                mad_ct = mad_cc

        # Define residual cutoffs
        maxres_cc = maxres_cross # absolute cutoff value
        maxres_ct = maxres_net # absolute cutoff value
        if maxres_cross >= 1:
            maxres_cc = mad_cc*maxres_cross
        if maxres_net >= 1:
            maxres_ct = mad_ct*maxres_net

        # Apply residual/offset dependednt weights to a priori weights
        nncc = 0
        nnct = 0
        ncc = 0
        nct = 0
        for i in range(0,ndt):
            if dt_idx[i]<=2:
                # Cross data
                ncc = ncc+1

                # bi ^5 offset weighting for cross data:
                # exp needs to be uneven so weights become
                # negative for offsets greater than x km
                if maxdcc!=-9:
                    dt_wt[i] = dt_wt[i]*(1-(dt_offs[i]/(maxdcc*1000))**5)**5

                # bi-cube residual weighting
                # needs to be a cube so that res > cutoff
                # became negative
                if maxres_cross>0 and dt_wt[i]>0.000001:
                    dt_wt[i] = dt_wt[i]*(1-(np.abs(dt_res[i])/maxres_cc)**3)**3
                if dt_wt[i]<minwght:
                    nncc = nncc+1
            else:
                # Catalog data
                nct = nct+1

                # bicube offset weighting for catalog data:
                if maxdct!=-9:
                    dt_wt[i] = dt_wt[i]*(1-(dt_offs[i]/(maxdct*1000))**3)**3

                # bi-cube residual weighting
                if dt_wt[i]>0.000001 and maxres_net>0:
                    dt_wt[i] = dt_wt[i]*(1-(np.abs(dt_res[i])/maxres_ct)**3)**3
                if dt_wt[i]<minwght:
                    nnct = nnct+1

        # check if neg residuals exist
        ineg = 0
        for j in range(0,ndt):
            if dt_wt[j]<minwght:
                #print('here ' + str(i) + '  ' + str(dt_wt[i]))
                ineg=1
                break

        if idata==1 or idata==3:
            log.write('cc res/dist cutoff: %7.3f s/ %5.2f km (%5.1f%%)\n' %
                      (maxres_cc,maxdcc,(nncc*100./ncc)))
        if idata==2 or idata==3:
            log.write('ct res/dist cutoff: %7.3f s/ %5.2f km (%5.1f%%)\n' %
                      (maxres_ct,maxdct,(nnct*100./nct)))

    if ineg==0:
        kiter=kiter+1

    return [ineg,dt_wt]


#@profile
def skip(log, kiter, minwght, ndt, nev, nsrc, nsta,
         ev_cusp, ev_date, ev_time, ev_mag, 
         ev_lat, ev_lon, ev_dep, ev_x, ev_y, ev_z,
         ev_herr, ev_zerr, ev_res,
         src_cusp, src_lat, src_lon, src_dep,
         src_lat0, src_lon0,
         src_x, src_y, src_z, src_t, src_x0, src_y0, src_z0, src_t0,
         sta_lab, sta_lat, sta_lon, #sta_dist, sta_az,
         sta_rmsc, sta_rmsn, sta_np, sta_ns, sta_nnp, sta_nns,
         dt_sta, dt_c1, dt_c2, dt_idx, dt_dt, dt_qual, dt_cal,
         dt_ista, dt_ic1, dt_ic2, dt_res, dt_wt, dt_offs,
         tmp_ttp, tmp_tts, tmp_xp, tmp_yp, tmp_zp, nct, ncc):
    """
    Skip outliers and air quakes
    """

    log.write('Skipping data...\n')
    # Skip data with large residuals
    #if kiter==1:
    ndtold = ndt
    nccold = 0
    nctold = 0
    ncc = 0
    nct = 0
    j = 0

    if kiter==1:
        nccold = (dt_idx<=2).sum()
        nctold = (dt_idx>2).sum()

    mask = (dt_wt>=minwght)
    dt_sta = dt_sta[mask]
    dt_c1 = dt_c1[mask]
    dt_c2 = dt_c2[mask]
    dt_idx = dt_idx[mask]
    dt_qual = dt_qual[mask]
    dt_dt = dt_dt[mask]
    dt_cal = dt_cal[mask]
    dt_res = dt_res[mask]
    dt_offs = dt_offs[mask]
    dt_wt = dt_wt[mask]

    ncc = (dt_idx<=2).sum()
    nct = (dt_idx>2).sum()
    ndt = ncc+nct

    log.write('# obs = %9i (%5.1f%%)\n' % (ndt,(ndt*100./ndtold)))
    if nccold>0. and nctold>0:
        log.write('# obs cc = %9i (%5.1f%%)\n' % (ncc,(ncc*100./nccold)))
        log.write('# obs ct = %9i (%5.1f%%)\n' % (nct,(nct*100./nctold)))

    # Skip events
    dt_ic1 = np.copy(dt_c1) # dt_ic1 is a workspace array
    dt_ic2 = np.copy(dt_c2) # dt_ic2 is a workspace array
    dt_ic1[0:ndt+1] = np.sort(dt_ic1[0:ndt+1])
    dt_ic2[0:ndt+1] = np.sort(dt_ic2[0:ndt+1])
    k = 0
    for i in range(0,nev):
        if ev_cusp[i] in dt_ic1 or ev_cusp[i] in dt_ic2:
            ev_date[k] = ev_date[i]
            ev_time[k] = ev_time[i]
            ev_cusp[k] = ev_cusp[i]
            ev_lat[k] = ev_lat[i]
            ev_lon[k] = ev_lon[i]
            ev_dep[k] = ev_dep[i]
            ev_mag[k] = ev_mag[i]
            ev_herr[k] = ev_herr[i]
            ev_zerr[k] = ev_zerr[i]
            ev_res[k] = ev_res[i]
            ev_x[k] = ev_x[i]
            ev_y[k] = ev_y[i]
            ev_z[k] = ev_z[i]
            k = k+1
    nev = k
    log.write('# events: %9i \n' % nev)

    # Skip sources
    # Uses sorted dt_ic1/2 arrays from above
    if nsrc!=1:
        k=0
        for i in range(0,nsrc):
            if src_cusp[i] in dt_ic1 or src_cusp[i] in dt_ic2:
                src_cusp[k] = src_cusp[i]
                src_lat[k] = src_lat[i]
                src_lon[k] = src_lon[i]
                src_lat0[k] = src_lat0[i]
                src_lon0[k] = src_lon0[i]
                src_dep[k] = src_dep[i]
                src_x[k] = src_x[i]
                src_y[k] = src_y[i]
                src_z[k] = src_z[i]
                src_t[k] = src_t[i]
                src_x0[k] = src_x0[i]
                src_y0[k] = src_y0[i]
                src_z0[k] = src_z0[i]
                src_t0[k] = src_t0[i]
                #for j in range(0,nsta):
                tmp_ttp[:,k] = tmp_ttp[:,i]
                tmp_tts[:,k] = tmp_tts[:,i]
                tmp_xp[:,k] = tmp_xp[:,i]
                tmp_yp[:,k] = tmp_yp[:,i]
                tmp_zp[:,k] = tmp_zp[:,i]
                k = k+1
        nsrc = k

    # Clean stations
    sta_itmp=np.zeros(nsta)
    #for j in range(0,ndt):
    #    for i in range(0,nsta):
    #        if dt_sta[j]==sta_lab[i]:
    #            sta_itmp[i] = 1
    #            break
    sta_itmp = np.where(np.in1d(sta_lab,dt_sta),1,0)
    k=0
    for i in range(0,nsta):
        if sta_itmp[i]==1:
            sta_lab[k] = sta_lab[i]
            sta_lat[k] = sta_lat[i]
            sta_lon[k] = sta_lon[i]
            #sta_dist[k] = sta_dist[i]
            #sta_az[k] = sta_az[i]
            sta_np[k] = sta_np[i]
            sta_ns[k] = sta_ns[i]
            sta_nnp[k] = sta_nnp[i]
            sta_nns[k] = sta_nns[i]
            sta_rmsc[k] = sta_rmsc[i]
            sta_rmsn[k] = sta_rmsn[i]
            #for j in range(0,nsrc):
            tmp_ttp[k,:] = tmp_ttp[i,:]
            tmp_tts[k,:] = tmp_tts[i,:]
            tmp_xp[k,:] = tmp_xp[i,:]
            tmp_yp[k,:] = tmp_yp[i,:]
            tmp_zp[k,:] = tmp_zp[i,:]
            k = k+1
    nsta = k
    log.write('# Stations = %9i \n' % nsta)

    # Index station labels and cuspids
    iicusp=np.argsort(ev_cusp[0:nev])
    icusp = np.zeros(nev)
    icusp[:] = ev_cusp[iicusp[:]]
    for i in range(0,ndt):
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

    return [ndt, nev, nsrc, nsta,
            ev_cusp, ev_date, ev_time, ev_mag, 
            ev_lat, ev_lon, ev_dep, ev_x, ev_y, ev_z,
            ev_herr, ev_zerr, ev_res,
            src_cusp, src_lat, src_lon, src_dep,
            src_lat0, src_lon0,
            src_x, src_y, src_z, src_t, src_x0, src_y0, src_z0, src_t0,
            sta_lab, sta_lat, sta_lon, #sta_dist, sta_az,
            sta_rmsc, sta_rmsn, sta_np, sta_ns, sta_nnp, sta_nns,
            dt_sta, dt_c1, dt_c2, dt_idx, dt_dt, dt_qual, dt_cal,
            dt_ista, dt_ic1, dt_ic2, dt_res, dt_wt, dt_offs,
            tmp_ttp, tmp_tts, tmp_xp, tmp_yp, tmp_zp, nct, ncc]


#@profile
def resstat(log, idata, ndt, nev, d, w, idx, 
            rms_cc, rms_ct, rms_cc0, rms_ct0,
            rms_ccold, rms_ctold, rms_cc0old, 
            rms_ct0old, dum):
    """
    Calculate residual statistics
    """
    # Get rms:
    rms_cc0old = rms_cc0
    rms_ct0old = rms_ct0
    rms_ccold = rms_cc
    rms_ctold = rms_ct 
    j=0
    sw_cc=0.
    sw_ct=0.
    for i in range(0,ndt):
        if idx[i]<=2:
            sw_cc = sw_cc + w[i]
            j = j+1
        else:
            sw_ct = sw_ct + w[i]
    f_cc=j/sw_cc        # Factor to scale weights for rms value
    f_ct=(ndt-j)/sw_ct  # Factor to scale weights for rms value

    rms_cc0=0.
    rms_ct0=0.
    av_cc0=0.
    av_ct0=0.
    rms_cc=0.
    rms_ct=0.
    av_cc=0.
    av_ct=0.
    j=0
    for i in range(0,ndt):
        if idx[i]<=2:
            rms_cc0 = rms_cc0 + d[i]**2
            av_cc0 = av_cc0 + d[i]
            rms_cc = rms_cc + (f_cc*w[i]*d[i])**2   # Weighted and scaled
            av_cc = av_cc + f_cc*w[i]*d[i]  # Weighted and scaled
            j = j+1
        else:
            rms_ct0 = rms_ct0 + d[i]**2
            av_ct0 = av_ct0 + d[i]
            rms_ct = rms_ct + (f_ct*w[i]*d[i])**2   # Weighted and scaled
            av_ct = av_ct + f_ct*w[i]*d[i]  # Weighted and scaled
    av_cc0 = av_cc0/j
    av_ct0 = av_ct0/(ndt-j)
    rms_cc0 = np.sqrt((rms_cc0-av_cc0**2/j)/(j-1))
    rms_ct0 = np.sqrt((rms_ct0-av_ct0**2/(ndt-j))/(ndt-j-1))
    av_cc = av_cc/j
    av_ct = av_ct/(ndt-j)
    rms_cc = np.sqrt((rms_cc-av_cc**2/j)/(j-1))
    rms_ct = np.sqrt((rms_ct-av_ct**2/(ndt-j))/(ndt-j-1))

    # More: residual average, rms, and variance:
    if (abs(dum+999)<0.0001):
        dav = 0.
        dvar = 0.
        dav1 = 0.
        dvar1 = 0.
    try:
        davold = dav
        dvarold = dvar
        dav1old = dav1 
        dvar1old = dvar1 
    except:
        davold = 0.
        dvarold = 0.
        dav1old = 0. 
        dvar1old = 0. 
    dav=0.   # unweighted
    dvar=0.  # unweighted
    dav1=0.  # unweighted
    dvar1=0. # unweighted
    sw=0.
    dav0=0.  # unweighted

    sw = np.sum(w)
    f = ndt/sw  # Factor to scale weights for rms value

    dav = np.sum(d) # unweighted
    dav0 = np.sum(w*d) # weighted
    dav1 = np.sum(f*w*d) # weighted and scaled
    dav = dav/ndt
    dav0 = dav0/ndt
    dav1 = dav1/ndt

    s = np.zeros(ndt)
    s1 = np.zeros(ndt)
    s = d*1000 - dav*1000 # in msec
    s1 = w*d*1000 - dav0*1000 # weighted in msec
    ss = np.sum(s)
    ss1 = np.sum(s1)
    dvar = np.sum(s*s)
    dvar1 = np.sum(s1*s1)
    if ndt>4*nev:
        dvar = (dvar - ss**2/ndt)/(ndt - 4*nev) # divide by number of degrees of freedom
        dvar1 = (dvar1 - ss1**2/ndt)/(ndt - 4*nev)
    else:
        dvar = (dvar/1) # divide by the number of degrees of freedom
        dvar1 = (dvar1/1)
        print('>>> Warning: ndt < 4*nev')
        log.write('>>> Warning: ndt < 4*nev \n')

    if (abs(dum+999)<0.0001): # original data
        log.write('Residual summary of initial data: \n')
        log.write(' absolute mean [s] = %7.4f \n' % dav)
        log.write(' weighted mean [s] = %7.4f \n' % dav1)
        log.write(' absolute variance [s] = %7.4f \n' % (dvar/1000))
        log.write(' weighted variance [s] = %7.4f \n' % (dvar1/1000))
        if idata==1 or idata==3:
            log.write(' absolute cc rms [s] = %7.4f \n' % rms_cc0)
            log.write(' weighted cc rms [s] (RMSCC) = %7.4f \n' % rms_cc)
        if idata==2 or idata==3:
            log.write(' absolute ct rms [s] = %7.4f \n' % rms_ct0)
            log.write(' weighted ct rms [s] (RMSCT) = %7.4f \n' % rms_ct)
    else:
        log.write('Residual summary: \n')
        #log.write(' absolute mean [s] = %7.4f (%7.2f %%) \n' % (dav,(dav-davold)*100/np.abs(davold)))
        #log.write(' weighted mean [s] = %7.4f (%7.2f %%) \n' % (dav1,(dav1-dav1old)*100/np.abs(davold)))
        #log.write(' absolute variance [s] = %10.4f (%7.2f %%) \n' % (dvar/1000,(dvar-dvarold)*100/dvarold))
        #log.write(' weighted variance [s] = %10.4f (%7.2f %%) \n' % (dvar1/1000,(dvar1-dvarold)*100/dvar1old))
        if idata==1 or idata==3:
            log.write(' absolute cc rms [s] = %7.4f (%7.2f %%) \n' % (rms_cc0,(rms_cc0-rms_cc0old)*100/rms_cc0old))
            log.write(' weighted cc rms [s] = %7.4f (%7.2f %%) \n' % (rms_cc,(rms_cc-rms_ccold)*100/rms_ccold))
        if idata==2 or idata==3:
            log.write(' absolute ct rms [s] = %7.4f (%7.2f %%) \n' % (rms_ct0,(rms_ct0-rms_ct0old)*100/rms_ct0old))
            log.write(' weighted ct rms [s] = %7.4f (%7.2f %%) \n' % (rms_ct,(rms_ct-rms_ctold)*100/rms_ctold))

    dum = dvar1

    return [rms_cc, rms_ct, rms_cc0, rms_ct0, rms_ccold, 
            rms_ctold, rms_cc0old, rms_ct0old, dum]            


#@profile
#@jit
def juliam(iyr,imo,idy,ihr,imn):
    kmo = np.array([0,31,59,90,120,151,181,212,243,273,304,334])
    leap = 1

    ky = iyr
    km = imo
    kd = idy
    if km<=0:
        km=1
    juliam = 365*ky
    kd = kmo[km]+kd
    ky4 = ky/4
    ky1 = ky/100
    ky0 = ky/1000
    kl = leap*(ky4-ky1+ky0)
    l = 0
    if ky4*4==ky and (ky1*100!=ky or ky0*1000==ky):
        l = leap
    if l!=0 and km<3:
        kl = kl-leap
    juliam = juliam+kd+kl
    juliam = juliam*24+ihr
    juliam = juliam*60+imn
    return juliam


#@profile
#@jit
def datum(itf,iyr,imo,idy,ihr,imn):
    kmo = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

    k = int(itf/60)
    imn = int(itf-k*60)
    kh = int(k/24)
    ihr = int(k-kh*24)
    iyr = int(kh/365)
    iskip=0
    while iskip==0:
        idd = int(kh-iyr*365)
        l = 0
        iyr4 = int(iyr/4)
        iyrh = int(iyr/100)
        iyrt = int(iyr/1000)
        ld = int(iyr4-iyrh+iyrt)
        if iyr4*4==iyr and (iyrh*100!=iyr or iyrt*1000==iyr):
            l = 1
        idd = idd-ld+l
        if idd>0:
            iskip=1
        else:
            if idd==0 and ihr==0 and imn==0:
                idy=0
                imo=0
            iyr = iyr-1
    kmo[1] = 28+l
    iskip=0
    for i in range(0,12):
        idd = idd - kmo[i]
        if idd<=0:
            iskip=1
            break
    if iskip==1:
        i = 11
    idy = idd+kmo[i]
    imo = i
    return [iyr,imo,idy,ihr,imn]




