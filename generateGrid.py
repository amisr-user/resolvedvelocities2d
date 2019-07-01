from constants import *
import numpy as np
import h5py
from pylab import *
import os.path

#this file generates grid and inst MATLAB analogues
def generateGrid(N, M, sLat, sLon, mlatR, mlonR, Dlat, Dlon):
    
    print "Building grid, size = " + str(N) + " x " + str(M)
    
    Nphi = N * M

    percentGridBuffer = 0.015 #set relative size of border region on grid, i.e. if percentGridBuffer = 0.02, the x axis of the grid will go from 0.98 min(mlonR) to 1.02 max(mlonR) and the y axis of the grid will go from 0.98 min(mlatR) to 1.02 max(mlatR)
    
    dx = ( (1.0 + percentGridBuffer) * np.nanmax(mlonR) - (1.0 - percentGridBuffer) * np.nanmin(mlonR)) / float(N) * Dlon
    dlon = dx / Dlon
    
    dy = ( (1.0 + percentGridBuffer) * np.nanmax(mlatR) - (1.0 - percentGridBuffer) * np.nanmin(mlatR)) / float(M) * Dlat
    dlat = dy / Dlat

    sLon = (1.0 - percentGridBuffer) * np.nanmin(mlonR)
    sLat = (1.0 - percentGridBuffer) * (np.nanmin(mlatR)) 


    x = np.empty((1, N))
    y = np.empty((1, M))

    for i in range(0, N):
        x[0, i] = sLon + (i + 1) * dlon   
    for i in range(0, M):
        y[0, i] = sLat + (i + 1) * dlat
        

    [X,Y] = np.meshgrid(x, y)

    XR = np.reshape(X, (Nphi, 1), order = 'F')
    YR = np.reshape(Y, (Nphi, 1), order = 'F')

    grid = {}
    grid['x'] = x
    grid['y'] = y
    grid['X'] = X
    grid['Y'] = Y
    grid['XR'] = XR
    grid['YR'] = YR
    grid['dx'] = dx
    grid['dy'] = dy
    grid['Nphi'] = Nphi
    grid['Nx'] = N
    grid['Ny'] = M
    grid['dlon'] = dlon
    grid['dlat'] = dlat

 
    return grid
    
    
def instrumentParams(pathToInstrumentFile):
     
    print "Getting instrument information. Reading parameters from: " + pathToInstrumentFile 
    
    if ('dat' not in locals()) and ('var' not in locals()):
        dat = h5py.File(pathToInstrumentFile)
        
    #unpack data
    beamCodes = np.asarray(dat['BeamCodes']).T
    altitude = np.asarray(dat['Geomag/Altitude']).T / 1.0e3
    rng = np.asarray(dat['Geomag/Range']).T / 1.0e3
    mlon = np.asarray(dat['Geomag/MagneticLongitude']).T
    mlat = np.asarray(dat['Geomag/MagneticLatitude']).T
    ke = np.asarray(dat['Geomag/ke']).T
    kn = np.asarray(dat['Geomag/kn']).T
    kz = np.asarray(dat['Geomag/kz']).T
    kpe = np.asarray(dat['Geomag/kpe']).T
    kpn = np.asarray(dat['Geomag/kpn']).T
    kap = np.asarray(dat['Geomag/kpar']).T
    Babs = np.asarray(dat['Geomag/Babs']).T
    siteLat = np.asarray(dat['Site/Latitude']).T
    siteMLat = np.asarray(dat['Site/MagneticLatitude']).T
    siteMLon = np.asarray(dat['Site/MagneticLongitude']).T
    time = np.asarray(dat['Time/dtime']).T
    time = np.mean(time, axis = 0)
   
    
    #beamcodes and time    
    Nbeams = beamCodes.shape[1]
    el = beamCodes[2, :]
    az = beamCodes[1, :]
    
    #lat and lon
    clat = siteLat * pi / 180; 
    Dlat = pi / 180 * a * (1-e2) / pow((sqrt(1 - e2 * pow(sin(clat),2))), 3); # m/deg
    Dlon = pi / 180 * a * cos(clat) / (sqrt(1 - e2 * pow(sin(clat),2))); # m/deg
    
    #geomag
    Nalt = altitude.shape[0]
    with np.errstate(invalid='ignore'):
        I = np.asarray(np.where(np.less(mlon, 200)))
        for i in range(I.shape[1]):
            mlon[I[0, i], I[1, i]] += 360.0
    
    Nmax = mlon.size
    #Beams = altitude / altitude * np.matlib.repmat(beamCodes[0, :], Nalt, 1)
    El = altitude / altitude * np.matlib.repmat(beamCodes[2, :], Nalt, 1)
    
    #reshape
    columnVectorShape = (Nalt * Nbeams, 1) 
    altR = np.reshape(altitude, columnVectorShape, order = 'F')
    rngR = np.reshape(rng, columnVectorShape, order = 'F')
    
    
    A = np.hstack((np.reshape(kpe, columnVectorShape, order = 'F'), np.reshape(kpn, columnVectorShape, order = 'F'),  np.reshape(kap, columnVectorShape, order = 'F')))
    
    #A = np.array([np.reshape(kpe, columnVectorShape, order = 'F'), np.reshape(kpn, columnVectorShape, order = 'F'), np.reshape(kap, columnVectorShape, order = 'F')])
    mlatR = np.reshape(mlat, columnVectorShape, order = 'F')
    mlonR = np.reshape(mlon, columnVectorShape, order = 'F')
    BabsR = np.reshape(Babs, columnVectorShape, order = 'F')
    #BeamsR = np.reshape(Beams, columnVectorShape, order = 'F')
    ElR = np.reshape(El, columnVectorShape, order = 'F')
    
    
    #make output dictionary
    instrument = {}
    instrument['altR'] = altR
    instrument['rngR'] = rngR
    instrument['A'] = A
    instrument['mlatR'] = mlatR
    instrument['mlonR'] = mlonR
    instrument['BabsR'] = BabsR
    instrument['ElR'] = ElR
    instrument['Nalt'] = Nalt
    instrument['Nbeams'] = Nbeams
    instrument['sLon'] = siteMLon
    instrument['sLat'] = siteMLat
    instrument['Dlat'] = Dlat
    instrument['Dlon'] = Dlon
    instrument['time'] = time
            
    return instrument
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    