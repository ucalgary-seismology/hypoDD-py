import numpy as np
from datetime import datetime
import hypoDD_functions as hf
import scipy.sparse.linalg as sp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
import sys


def lsfit_svd(log, iteri, ndt, nev, nsrc, damp, mod_ratio,
              idata, ev_cusp, src_cusp,
              dt_res, dt_wt, dt_ista, dt_ic1, dt_ic2,
              exav, eyav, ezav, etav, dxav, dyav, dzav, dtav,   
              rms_cc, rms_ct, rms_cc0, rms_ct0,
              rms_ccold, rms_ctold, rms_cc0old, rms_ct0old,
              tmp_xp, tmp_yp, tmp_zp, dt_idx):
    """
    Direct fortran to python conversion of lsfit_svd.f from hypoDD v1.3

    Note: Use instead updated/faster version of this function --> fast_svd
    """    

    # Set up full G matrix
    g = np.zeros((ndt,4*nev))

    for i in range(0,ndt):
        if nsrc==1:
            k1=0
            k2=0
        else:
            k1=int(dt_ic1[i])
            k2=int(dt_ic2[i])
        if int(dt_idx[i])==2 or int(dt_idx[i])==4:
            g[i,(4*int(dt_ic1[i]))] = tmp_xp[int(dt_ista[i]),k1]*mod_ratio
            g[i,(4*int(dt_ic1[i])+1)] = tmp_yp[int(dt_ista[i]),k1]*mod_ratio
            g[i,(4*int(dt_ic1[i])+2)] = tmp_zp[int(dt_ista[i]),k1]*mod_ratio
            g[i,(4*int(dt_ic1[i])+3)] = 1.
            g[i,(4*int(dt_ic2[i]))] = -tmp_xp[int(dt_ista[i]),k2]*mod_ratio
            g[i,(4*int(dt_ic2[i])+1)] = -tmp_yp[int(dt_ista[i]),k2]*mod_ratio
            g[i,(4*int(dt_ic2[i])+2)] = -tmp_zp[int(dt_ista[i]),k2]*mod_ratio
            g[i,(4*int(dt_ic2[i])+3)] = -1.
        elif int(dt_idx[i])==1 or int(dt_idx[i])==3:
            g[i,(4*int(dt_ic1[i]))] = tmp_xp[int(dt_ista[i]),k1]
            g[i,(4*int(dt_ic1[i])+1)] = tmp_yp[int(dt_ista[i]),k1]
            g[i,(4*int(dt_ic1[i])+2)] = tmp_zp[int(dt_ista[i]),k1]
            g[i,(4*int(dt_ic1[i])+3)] = 1.
            g[i,(4*int(dt_ic2[i]))] = -tmp_xp[int(dt_ista[i]),k2]
            g[i,(4*int(dt_ic2[i])+1)] = -tmp_yp[int(dt_ista[i]),k2]
            g[i,(4*int(dt_ic2[i])+2)] = -tmp_zp[int(dt_ista[i]),k2]
            g[i,(4*int(dt_ic2[i])+3)] = -1.

    # Weight data
    wtinv = np.zeros(ndt)
    d = np.zeros(ndt)
    wt = np.zeros(ndt)
    wt = dt_wt[0:ndt]
    d = dt_res[0:ndt]*1000.*wt # data in ms so results are in m
    for i in range(0,ndt):
        # wt must be same as dt_wt but with different dimensions
        # should not be changed so statistics in resstat will not be messed up
        if wt[i]!=0:
            wtinv[i] = 1./wt[i]
        else:
            wtinv[i] = 1.

    # Weight G matrix
    # multiply 2d matrix by 1d
    for i in range(0,4*nev):
        g[:,i] = wt*g[:,i]

    # Add four extra rows to make mean shift zero
    # This should make the design matrix non-singular
    temp = np.zeros(4*nev)
    for i in range(0,4):
        d = np.append(d,[0.])
        wt = np.append(wt,[1.])
        g = np.append(g,[temp],axis=0)
        for j in range(0,nev):
            g[ndt+i,4*j-4+i] = 1.0
    nndt = ndt+4

    # Column scaling
    norm = np.zeros(4*nev)
    for j in range(0,4*nev):
        for i in range(0,nndt):
            norm[j] = norm[j]+g[i,j]**2
    for j in range(0,4*nev):
        norm[j] = np.sqrt(norm[j]/nndt)
    for j in range(0,4*nev):
        for i in range(0,nndt):
            g[i,j] = g[i,j]/norm[j]

    # Testing
    norm_test = np.zeros(4*nev,dtype='float32')
    for j in range(0,4*nev):
        for i in range(0,nndt):
            norm_test[j] = norm_test[j] + g[i,j]**2
        norm_test[j] = np.sqrt(norm_test[j]/nndt)
        if np.abs(norm_test[j]-1.) > 0.001:
            raise Exception('Fatal Error: SVD G scaling.')

    # Run SVD
    log.write('~ singular value decomposition of G %6i x %6i matrix ... \n\n ' % (nndt,nev*4))
    [u,q,v] = np.linalg.svd(g)
    v = np.transpose(v)

    # Check for singular values near zero
    log.write('~ backsubstitution ... \n')
    izero = 0
    qmax = 0.0
    for i in range(0,4*nev):
        if q[i]<0.:
            raise Exception('Fatal Error: (svd: neg singular value)')
        if q[i]>qmax:
            qmax = q[i]
    qmin = qmax*0.0000001
    for i in range(0,4*nev):
        if q[i]<qmin:
            q[i] = 0.
            izero = izero+1
    if izero>0:
        log.write('>>> %3i singular values close/equal to zero \n' % izero)
        print('>>> %3i singular values close/equal to zero' % izero)

    # Backsubstitution (get x' from Gx=d: x=v*diag(1/q)*t(u)*d)
    # Compute diag(1/q)*t(u)*d
    tmp = np.zeros(4*nev)
    for j in range(0,4*nev):
        s = 0.
        if q[j]!=0.:
            for i in range(0,nndt):
                s = s+u[i,j]*d[i]
            s = s/q[j]
        tmp[j] = s

    x = np.zeros(4*nev)
    # Multiply by V
    for i in range(0,4*nev):
        s = 0.
        for j in range(0,4*nev):
            s = s + v[i,j]*tmp[j]
        x[i] = s

    # Rescale model vector and G
    for j in range(0,4*nev):
        x[j] = x[j]/norm[j]
        for i in range(0,ndt):
            g[i,j] = g[i,j]*norm[j]

    # Unweight G matrix
    for i in range(0,4*nev):
        g[0:ndt,i] = wtinv*g[0:ndt,i]

    # Predict data from g*x', get residuals
    dd = np.zeros(ndt)
    for i in range(0,ndt):
        s = 0.
        for j in range(0,4*nev):
            s = s + g[i,j]* x[j]
        dd[i] = s
    for i in range(0,ndt):
        dt_res[i] = dt_res[i] - dd[i]/1000.
    
    # Get covariance matrix: cvm = v*(1/q**2)*vt
    cvm = np.zeros((4*nev,4*nev))
    for i in range(0,4*nev):
        for j in range(0,i+1):
            s = 0.
            for k in range(0,4*nev):
                if q[k] != 0.:
                    s = s + (v[i,k]*v[j,k])*(1./(q[k]*q[k]))
            cvm[i,j] = s
            cvm[j,i] = s

    # Get residual statistics
    resvar1 = 0.
    [rms_cc,rms_ct,rms_cc0,rms_ct0, rms_ccold,rms_ctold,
     rms_cc0old,rms_ct0old,resvar1] = hf.resstat(log,idata,ndt,nev,dt_res,dt_wt,dt_idx,
                                                 rms_cc,rms_ct,rms_cc0,rms_ct0,
                                                 rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,
                                                 resvar1)

    # Errors for 95% confidence level
    factor=1.96
    se = np.zeros(4*nev)
    for i in range(0,4*nev):
        se[i] = np.sqrt(cvm[i,i])*np.sqrt(resvar1)*factor

    # Rescale Errors
    for i in range(0,4*nev):
        se[i] = se[i]/norm[i]

    # Store solution and errors
    src_dx = np.zeros(nev)
    src_dy = np.zeros(nev)
    src_dz = np.zeros(nev)
    src_dt = np.zeros(nev)
    src_ex = np.zeros(nev)
    src_ey = np.zeros(nev)
    src_ez = np.zeros(nev)
    src_et = np.zeros(nev)
    for i in range(0,nev):
        src_dx[i] = -x[4*i]
        src_dy[i] = -x[4*i+1]
        src_dz[i] = -x[4*i+2]
        src_dt[i] = -x[4*i+3]
        src_ex[i] = se[4*i]
        src_ey[i] = se[4*i+1]
        src_ez[i] = se[4*i+2]
        src_et[i] = se[4*i+3]
        src_cusp[i] = ev_cusp[i]

    # Output statistics
    exavold = exav
    eyavold = eyav
    ezavold = ezav 
    etavold = etav 
    dxavold = dxav 
    dyavold = dyav 
    dzavold = dzav 
    dtavold = dtav 
    exav = 0.
    eyav = 0.
    ezav = 0.
    etav = 0.
    dxav = 0.
    dyav = 0.
    dzav = 0.
    dtav = 0.
    for i in range(0,nev):
        exav = exav + src_ex[i]
        eyav = eyav + src_ey[i]
        ezav = ezav + src_ez[i]
        etav = etav + src_et[i]
        dxav = dxav + np.abs(src_dx[i])
        dyav = dyav + np.abs(src_dy[i])
        dzav = dzav + np.abs(src_dz[i])
        dtav = dtav + np.abs(src_dt[i])
    exav = exav/nev
    eyav = eyav/nev
    ezav = ezav/nev
    etav = etav/nev
    dxav = dxav/nev
    dyav = dyav/nev
    dzav = dzav/nev
    dtav = dtav/nev

    if iteri==1:
        exavold = exav
        eyavold = eyav
        ezavold = ezav
        etavold = etav
        dxavold = dxav
        dyavold = dyav
        dzavold = dzav
        dtavold = dtav

    log.write('Location summary: \n')
    log.write('mean 2sig-error (x,y,z,t) [m,ms]: %7.1f %7.1f %7.1f %7.1f \n' % (exav,eyav,ezav,etav))
    log.write('  ( %7.1f %7.1f %7.1f %7.1f )  \n' % (exav-exavold,eyav-eyavold,ezav-ezavold,etav-etavold))
    log.write('mean shift (x,y,z,t) [m,ms] (DX,DY,DZ,DT): %7.1f %7.1f %7.1f %7.1f \n' % (dxav,dyav,dzav,dtav))
    log.write('  ( %7.1f %7.1f %7.1f %7.1f )  \n' % (dxav-dxavold,dyav-dyavold,dzav-dzavold,dtav-dtavold))

    return  [src_cusp,src_dx,src_dy,src_dz,src_dt, 
             src_ex,src_ey,src_ez,src_et,
             exav,eyav,ezav,etav,dxav,dyav,dzav,dtav,
             rms_cc,rms_ct,rms_cc0,rms_ct0,
             rms_ccold,rms_ctold,rms_cc0old,rms_ct0old]


#@profile
def fast_svd(log, iteri, ndt, nev, nsrc, damp, mod_ratio,
             idata, ev_cusp, src_cusp,
             dt_res, dt_wt, dt_ista, dt_ic1, dt_ic2,
             exav, eyav, ezav, etav, dxav, dyav, dzav, dtav,   
             rms_cc, rms_ct, rms_cc0, rms_ct0,
             rms_ccold, rms_ctold, rms_cc0old, rms_ct0old,
             tmp_xp, tmp_yp, tmp_zp, dt_idx):
    """
    Translation of lsfit_svd.f
    """
    # Set up full G matrix    
    g = np.zeros((ndt+4,4*nev))
    dt_ista = dt_ista.astype('int')
    dt_ic1 = dt_ic1.astype('int')
    dt_ic2 = dt_ic2.astype('int')
    
    if nsrc==1:
        for i in range(0,ndt):
            k3 = 4*dt_ic1[i]
            k4 = 4*dt_ic2[i]

            g[i,k3] = tmp_xp[dt_ista[i],0]
            g[i,k3+1] = tmp_yp[dt_ista[i],0]
            g[i,k3+2] = tmp_zp[dt_ista[i],0]
            g[i,k3+3] = 1.
            g[i,k4] = -tmp_xp[dt_ista[i],0]
            g[i,k4+1] = -tmp_yp[dt_ista[i],0]
            g[i,k4+2] = -tmp_zp[dt_ista[i],0]
            g[i,k4+3] = -1.
    else:
        
        k1 = dt_ic1
        k2 = dt_ic2
        k3 = 4*k1
        k4 = 4*k2
        for i in range(0,ndt):
            g[i,k3[i]] = tmp_xp[dt_ista[i],k1[i]]
            g[i,k3[i]+1] = tmp_yp[dt_ista[i],k1[i]]
            g[i,k3[i]+2] = tmp_zp[dt_ista[i],k1[i]]
            g[i,k3[i]+3] = 1.
            g[i,k4[i]] = -tmp_xp[dt_ista[i],k2[i]]
            g[i,k4[i]+1] = -tmp_yp[dt_ista[i],k2[i]]
            g[i,k4[i]+2] = -tmp_zp[dt_ista[i],k2[i]]
            g[i,k4[i]+3] = -1.

    g[0:ndt,:] = np.where(np.logical_or(dt_idx[0:ndt].reshape((ndt,1))==2,dt_idx[0:ndt].reshape((ndt,1))==4),g[0:ndt,:]*mod_ratio,g[0:ndt,:])
   
    # Weight data
    wt = np.zeros((ndt+4))
    wtinv = np.zeros((ndt))
    d = np.zeros((ndt+4))
    wt[0:ndt] = np.copy(dt_wt[0:ndt])
    d[0:ndt] = dt_res[0:ndt]*1000.*wt[0:ndt] # data in ms so results are in m
    wtinv = np.copy(wt[0:ndt])
    wtinv = np.where(wtinv==0.,1.,1./wtinv)
    # Weight G matrix
    # multiply 2d matrix by 1d
    g[0:ndt] = wt[0:ndt].reshape((ndt,1))*g[0:ndt,:]

    # Add four extra rows to make mean shift zero
    # This should make the design matrix non-singular
    wt[ndt:] = 1.
    g[ndt,0:4*nev-3:4] = 1.0
    g[ndt+1,1:4*nev-2:4] = 1.0
    g[ndt+2,2:4*nev-1:4] = 1.0
    g[ndt+3,3:4*nev:4] = 1.0
    nndt = ndt+4

    # Column scaling
    norm = np.zeros((4*nev))
    norm = np.sqrt(np.sum(g*g,axis=0))/nndt
    g = g/norm

    # Run SVD
    log.write('~ singular value decomposition of G %6i x %6i matrix ... \n\n ' % (nndt,nev*4))
    [u,q,v] = np.linalg.svd(g,full_matrices=False)
    v = np.transpose(v)

    # Check for singular values near zero
    log.write('~ backsubstitution ... \n')
    izero = 0
    qmax = np.amax(q)
    q_tmp = np.where(q>0,False,True)
    if q_tmp.any():
        raise Exception('Fatal Error: (svd: neg singular value)')
    qmin = qmax*0.0000001
    izero = (q<qmin).sum()
    q = np.where(q<qmin,0.,q)
    izero = (q<qmin).sum()
    if izero>0:
        log.write('>>> %3i singular values close/equal to zero \n' % izero)
        print('>>> %3i singular values close/equal to zero' % izero)

    # Backsubstitution (get x' from Gx=d: x=v*diag(1/q)*t(u)*d)
    # Compute diag(1/q)*t(u)*d
    tmp = np.zeros(4*nev)
    for j in range(0,4*nev):
        if q[j]!=0.:
            tmp[j] = np.sum(u[:,j]*d/q[j])
    x = np.sum(v*tmp,axis=1)

    # Rescale model vector and G
    x = x/norm
    g = g*norm

    # Predict data from g*x', get residuals
    dd = np.sum(g[0:ndt,:]*x,axis=1)
    dt_res[0:ndt] = dt_res[0:ndt] - dd/1000.
    
    # Get covariance matrix: cvm = v*(1/q**2)*vt
    q2 = 1./(q*q)
    cvm = np.zeros((4*nev,4*nev))
    for i in range(0,4*nev):
        for j in range(0,i+1):
            s = np.sum(np.where(q!=0.,v[i,:]*v[j,:]*q2,0.))
            cvm[i,j] = s
            cvm[j,i] = s

    # Get residual statistics
    resvar1 = 0.
    [rms_cc,rms_ct,rms_cc0,rms_ct0, rms_ccold,rms_ctold,
     rms_cc0old,rms_ct0old,resvar1] = hf.resstat(log,idata,ndt,nev,dt_res,dt_wt,dt_idx,
                                                 rms_cc,rms_ct,rms_cc0,rms_ct0,
                                                 rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,
                                                 resvar1)

    # Errors for 95% confidence level
    factor=1.96*np.sqrt(resvar1)
    se = np.zeros(4*nev)
    for i in range(0,4*nev):
        se[i] = np.sqrt(cvm[i,i])*factor
    # Rescale Errors
    se = se/norm

    # Store solution and errors
    src_dx = np.zeros(nev)
    src_dy = np.zeros(nev)
    src_dz = np.zeros(nev)
    src_dt = np.zeros(nev)
    src_ex = np.zeros(nev)
    src_ey = np.zeros(nev)
    src_ez = np.zeros(nev)
    src_et = np.zeros(nev)
    src_dx = -x[0:4*nev-3:4]
    src_dy = -x[1:4*nev-2:4]
    src_dz = -x[2:4*nev-1:4]
    src_dt = -x[3:4*nev:4]
    src_ex = se[0:4*nev-3:4]
    src_ey = se[1:4*nev-2:4]
    src_ez = se[2:4*nev-1:4]
    src_et = se[3:4*nev:4]
    src_cusp = np.copy(ev_cusp)

    # Output statistics
    exavold = exav
    eyavold = eyav
    ezavold = ezav 
    etavold = etav 
    dxavold = dxav 
    dyavold = dyav 
    dzavold = dzav 
    dtavold = dtav 
    exav = np.sum(src_ex)/nev
    eyav = np.sum(src_ey)/nev
    ezav = np.sum(src_ez)/nev
    etav = np.sum(src_et)/nev
    dxav = np.sum(np.abs(src_dx))/nev
    dyav = np.sum(np.abs(src_dy))/nev
    dzav = np.sum(np.abs(src_dz))/nev
    dtav = np.sum(np.abs(src_dt))/nev
    if iteri==1:
        exavold = exav
        eyavold = eyav
        ezavold = ezav
        etavold = etav
        dxavold = dxav
        dyavold = dyav
        dzavold = dzav
        dtavold = dtav

    log.write('Location summary: \n')
    log.write('mean 2sig-error (x,y,z,t) [m,ms]: %7.1f %7.1f %7.1f %7.1f \n' % (exav,eyav,ezav,etav))
    log.write('  ( %7.1f %7.1f %7.1f %7.1f )  \n' % (exav-exavold,eyav-eyavold,ezav-ezavold,etav-etavold))
    log.write('mean shift (x,y,z,t) [m,ms] (DX,DY,DZ,DT): %7.1f %7.1f %7.1f %7.1f \n' % (dxav,dyav,dzav,dtav))
    log.write('  ( %7.1f %7.1f %7.1f %7.1f )  \n' % (dxav-dxavold,dyav-dyavold,dzav-dzavold,dtav-dtavold))

    return  [src_cusp,src_dx,src_dy,src_dz,src_dt, 
             src_ex,src_ey,src_ez,src_et,
             exav,eyav,ezav,etav,dxav,dyav,dzav,dtav,
             rms_cc,rms_ct,rms_cc0,rms_ct0,
             rms_ccold,rms_ctold,rms_cc0old,rms_ct0old]


#@profile
def lsfit_lsqr(log, iter, ndt, nev, nsrc,
               damp, mod_ratio, 
               idata, ev_cusp, src_cusp,
               dt_res, dt_wt,
               dt_ista, dt_ic1, dt_ic2,
               exav, eyav, ezav, etav, dxav, dyav, dzav, dtav,
               rms_cc, rms_ct, rms_cc0, rms_ct0,
               rms_ccold, rms_ctold, rms_cc0old, rms_ct0old,
               tmp_xp, tmp_yp, tmp_zp, dt_idx):

    log.write('~ Setting up G matrix: \n')

    nar = 8*ndt
    nndt = ndt
    row_i = np.zeros((nar),dtype=int)
    col_i = np.zeros((nar),dtype=int)
    rw = np.zeros(nar)
    wt = np.copy(dt_wt[0:ndt])
    wtinv = np.zeros(ndt)

    d = dt_res[0:ndt]*1000.*wt[0:ndt]
    dt_ista = dt_ista.astype(int)
    wtinv = np.where(wt==0.,1.,1./wt)
    
    # Set up row and column indexes and data input for sparse a matrix
    row_i[0:ndt] = np.arange(ndt)
    row_i[ndt:2*ndt] = np.arange(ndt)
    row_i[2*ndt:3*ndt] = np.arange(ndt)
    row_i[3*ndt:4*ndt] = np.arange(ndt)
    row_i[4*ndt:5*ndt] = np.arange(ndt)
    row_i[5*ndt:6*ndt] = np.arange(ndt)
    row_i[6*ndt:7*ndt] = np.arange(ndt)
    row_i[7*ndt:8*ndt] = np.arange(ndt)

    if nsrc==1:
        k1 = int(0)
        k2 = int(0)

        rw[0:ndt]       = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),tmp_xp[dt_ista[0:ndt],k1]*wt[0:ndt]*mod_ratio,tmp_xp[dt_ista[0:ndt],k1]*wt[0:ndt])
        rw[ndt:2*ndt]   = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),tmp_yp[dt_ista[0:ndt],k1]*wt[0:ndt]*mod_ratio,tmp_yp[dt_ista[0:ndt],k1]*wt[0:ndt])
        rw[2*ndt:3*ndt] = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),tmp_zp[dt_ista[0:ndt],k1]*wt[0:ndt]*mod_ratio,tmp_zp[dt_ista[0:ndt],k1]*wt[0:ndt])
        rw[3*ndt:4*ndt] = wt[0:ndt]
        rw[4*ndt:5*ndt] = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),-tmp_xp[dt_ista[0:ndt],k2]*wt[0:ndt]*mod_ratio,-tmp_xp[dt_ista[0:ndt],k2]*wt[0:ndt])
        rw[5*ndt:6*ndt] = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),-tmp_yp[dt_ista[0:ndt],k2]*wt[0:ndt]*mod_ratio,-tmp_yp[dt_ista[0:ndt],k2]*wt[0:ndt])
        rw[6*ndt:7*ndt] = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),-tmp_zp[dt_ista[0:ndt],k2]*wt[0:ndt]*mod_ratio,-tmp_zp[dt_ista[0:ndt],k2]*wt[0:ndt])
        rw[7*ndt:8*ndt] = -wt[0:ndt]

        # Set up column indexes with non-zero elements
        col_i[0:ndt] = 4*k1
        col_i[ndt:2*ndt] = 4*k1+1
        col_i[2*ndt:3*ndt] = 4*k1 + 2
        col_i[3*ndt:4*ndt] = 4*k1 + 3
        col_i[4*ndt:5*ndt] = 4*k2 
        col_i[5*ndt:6*ndt] = 4*k2 + 1
        col_i[6*ndt:7*ndt] = 4*k2 + 2
        col_i[7*ndt:8*ndt] = 4*k2 + 3

    else:
        k1 = dt_ic1[0:ndt].astype(int)
        k2 = dt_ic2[0:ndt].astype(int)

        rw[0:ndt]       = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),tmp_xp[dt_ista[0:ndt],k1[0:ndt]]*wt[0:ndt]*mod_ratio,tmp_xp[dt_ista[0:ndt],k1[0:ndt]]*wt[0:ndt])
        rw[ndt:2*ndt]   = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),tmp_yp[dt_ista[0:ndt],k1[0:ndt]]*wt[0:ndt]*mod_ratio,tmp_yp[dt_ista[0:ndt],k1[0:ndt]]*wt[0:ndt])
        rw[2*ndt:3*ndt] = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),tmp_zp[dt_ista[0:ndt],k1[0:ndt]]*wt[0:ndt]*mod_ratio,tmp_zp[dt_ista[0:ndt],k1[0:ndt]]*wt[0:ndt])
        rw[3*ndt:4*ndt] = wt[0:ndt]
        rw[4*ndt:5*ndt] = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),-tmp_xp[dt_ista[0:ndt],k2[0:ndt]]*wt[0:ndt]*mod_ratio,-tmp_xp[dt_ista[0:ndt],k2[0:ndt]]*wt[0:ndt])
        rw[5*ndt:6*ndt] = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),-tmp_yp[dt_ista[0:ndt],k2[0:ndt]]*wt[0:ndt]*mod_ratio,-tmp_yp[dt_ista[0:ndt],k2[0:ndt]]*wt[0:ndt])
        rw[6*ndt:7*ndt] = np.where((dt_idx[0:ndt]==2)|(dt_idx[0:ndt]==4),-tmp_zp[dt_ista[0:ndt],k2[0:ndt]]*wt[0:ndt]*mod_ratio,-tmp_zp[dt_ista[0:ndt],k2[0:ndt]]*wt[0:ndt])
        rw[7*ndt:8*ndt] = -wt[0:ndt]

        # Set up column indexes with non-zero elements
        col_i[0:ndt] = 4*k1
        col_i[ndt:2*ndt] = 4*k1+1
        col_i[2*ndt:3*ndt] = 4*k1 + 2
        col_i[3*ndt:4*ndt] = 4*k1 + 3
        col_i[4*ndt:5*ndt] = 4*k2 
        col_i[5*ndt:6*ndt] = 4*k2 + 1
        col_i[6*ndt:7*ndt] = 4*k2 + 2
        col_i[7*ndt:8*ndt] = 4*k2 + 3

    # Scale G matrix so the L2 norm of each column is 1
    log.write('~ Scaling G Columns \n')

    norm = np.zeros(4*nev)
    # G array scaling
    for i in range(0,nar):
        norm[col_i[i]] = norm[col_i[i]] + rw[i]*rw[i]
    norm = np.sqrt(norm/nndt)
    rw[:] = rw[:]/norm[col_i[:]]

    # Least square fitting using the algorithm of Paige and Saunders 1982
    # Set up input parameter first
    m = nndt
    n = 4*nev
    leniw = nar
    lenrw = nar

    w1 = np.zeros(4*nev)
    w2 = np.zeros(4*nev)
    x = np.zeros(4*nev)
    se = np.zeros(4*nev)
    atol = 0.000001
    btol = 0.000001
    conlim = 100000.0
    itnlim = 100*n
    istop = 0
    anorm = 0.0
    acond = 0.0
    rnorm = 0.0
    arnorm = 0.0
    xnorm = 0.0

    datet = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    log.write('~ lsqr .... %s' % datet)

    # Build a --- pulled from lsqr.f in hypoDD v1.3
    a = csc_matrix((rw,(row_i,col_i)),shape=(nndt,4*nev))

    # Call lsqr
    [x,istop,itn,r1norm,r2norm,anorm,
     acond,arnorm,xnorm,var] = sp.lsqr(a,d,damp,atol,btol,conlim,itnlim,calc_var=True)
    log.write("  istop = %1i; acond (CND)= %8.1f; anorm = %8.1f; arnorm = %8.1f; xnorm = %8.1f \n" % 
              (istop, acond, anorm, arnorm, xnorm))

    # Calculate se --- pulled form lsqr.f in hypoDD v1.3
    se = r2norm*np.sqrt(var/nndt)
    

    if nsrc==1:
        nsrc = nev
    # Rescale model vector
    x = x/norm
    se = se/norm

    # Unweight and rescale G matrix
    rw[0:ndt] = rw[0:ndt]*wtinv[0:ndt]*norm[col_i[0:ndt]]
    rw[ndt:2*ndt] = rw[ndt:2*ndt]*wtinv[0:ndt]*norm[col_i[0:ndt]]
    rw[2*ndt:3*ndt] = rw[2*ndt:3*ndt]*wtinv[0:ndt]*norm[col_i[0:ndt]]
    rw[3*ndt:4*ndt] = rw[3*ndt:4*ndt]*wtinv[0:ndt]*norm[col_i[0:ndt]]
    rw[4*ndt:5*ndt] = rw[4*ndt:5*ndt]*wtinv[0:ndt]*norm[col_i[0:ndt]]
    rw[5*ndt:6*ndt] = rw[5*ndt:6*ndt]*wtinv[0:ndt]*norm[col_i[0:ndt]]
    rw[6*ndt:7*ndt] = rw[6*ndt:7*ndt]*wtinv[0:ndt]*norm[col_i[0:ndt]]
    rw[7*ndt:8*ndt] = rw[7*ndt:8*ndt]*wtinv[0:ndt]*norm[col_i[0:ndt]]

    # Compute residuals from d = G*x
    d = -1*np.copy(dt_res)*1000.0
    d[row_i[:]] = d[row_i[:]] + rw[:]*x[col_i[:]]
    dt_res = -1*d/1000.0

    # Get residual statistics (avrg, rms, var..)
    resvar1 = 0.
    [rms_cc,rms_ct,rms_cc0,rms_ct0, rms_ccold,rms_ctold,
     rms_cc0old,rms_ct0old,resvar1] = hf.resstat(log,idata,ndt,nev,dt_res,dt_wt,dt_idx,
                                                 rms_cc,rms_ct,rms_cc0,rms_ct0,
                                                 rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,
                                                 resvar1)

    # Scale errors
    # The standard error estimates returned by LSQR increase monotonically
    # with the iterations.  If LSQR shuts down early because of loose tolerances,
    # or because the rhs-vector is special, the estimates will be too small.

    # Remember that se(j) is covariance(j) / (m - n)
    # where m - n = 1000000.  I've never quite understood why we
    # divide by that number.

    # Errors for the 95% confidence level,
    # thus multiply the standard errors by 2.7955
    factor = 2.7955

    # Store solution and errors
    src_dx = np.zeros(nev)
    src_dy = np.zeros(nev)
    src_dz = np.zeros(nev)
    src_dt = np.zeros(nev)
    src_ex = np.zeros(nev)
    src_ey = np.zeros(nev)
    src_ez = np.zeros(nev)
    src_et = np.zeros(nev)
    src_dx = -x[0:4*nev-3:4]
    src_dy = -x[1:4*nev-2:4]
    src_dz = -x[2:4*nev-1:4]
    src_dt = -x[3:4*nev:4]
    src_ex = np.sqrt(se[0:4*nev-3:4])*np.sqrt(resvar1)*factor
    src_ey = np.sqrt(se[1:4*nev-2:4])*np.sqrt(resvar1)*factor
    src_ez = np.sqrt(se[2:4*nev-1:4])*np.sqrt(resvar1)*factor
    src_et = np.sqrt(se[3:4*nev:4])*np.sqrt(resvar1)*factor
    src_cusp = np.copy(ev_cusp)

    #Get average errors and vector changes
    exavold = exav
    eyavold = eyav
    ezavold = ezav
    etavold = etav
    dxavold = dxav
    dyavold = dyav
    dzavold = dzav
    dtavold = dtav
    exav = np.sum(src_ex)/nev
    eyav = np.sum(src_ey)/nev
    ezav = np.sum(src_ez)/nev
    etav = np.sum(src_et)/nev
    dxav = np.sum(np.abs(src_dx))/nev
    dyav = np.sum(np.abs(src_dy))/nev
    dzav = np.sum(np.abs(src_dz))/nev
    dtav = np.sum(np.abs(src_dt))/nev

    if iter==1:
        exavold = exav
        eyavold = eyav
        ezavold = ezav
        etavold = etav
        dxavold = dxav
        dyavold = dyav
        dzavold = dzav
        dtavold = dtav

    # Output location statistics
    log.write('Location summary: \n')
    log.write(' mean 2sig-error (x,y,z,t) [m,ms]: %7.1f, %7.1f, %7.1f, %7.1f, ( %7.1f, %7.1f, %7.1f, %7.1f), \n' %
              (exav, eyav, ezav, etav, exav-exavold, eyav-eyavold, ezav-ezavold, etav-etavold))
    log.write(' mean shift (x,y,z,t) [m,ms] (DX,DY,DZ,DT): %7.1f, %7.1f, %7.1f, %7.1f, ( %7.1f, %7.1f, %7.1f, %7.1f), \n' %
              (dxav, dyav, dzav, dtav, dxav-dxavold, dyav-dyavold, dzav-dzavold, dtav-dtavold))

    return [src_cusp,src_dx,src_dy,src_dz,src_dt, 
            src_ex,src_ey,src_ez,src_et,
            exav,eyav,ezav,etav,dxav,dyav,dzav,dtav,
            rms_cc,rms_ct,rms_cc0,rms_ct0,
            rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,acond]




