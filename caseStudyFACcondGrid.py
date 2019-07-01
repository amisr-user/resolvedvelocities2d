#reads in height integrated SigmaH and SigmaP conductances for 2012-11-07 stable arc + substorm event, plus their magnetic coordinates. 
#



import scipy.io
import numpy as np
import matplotlib
import pylab as plt
from scipy import interpolate
from constants import *
import numpy as np
import h5py
from pylab import *
import os.path

np.set_printoptions(threshold=np.nan)


#this file generates grid and inst MATLAB analogues
def getGradientsFitGrid(X, Y):
    inst = instrumentParams("/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/data sources (radar and optical)/pfisr radar data/20121107.001_lp_5min-cal/20121107.001_lp_5min-cal.h5")
    grid = generateGrid(X.shape[1], Y.shape[0], inst["sLat"], inst["sLon"], inst["mlatR"], inst["mlonR"], inst["Dlat"], inst["Dlon"])
    g1 = computeGradientOperators(X.shape[1], Y.shape[0])
    dx = np.asarray(grid["dx"])
    dy = np.asarray(grid["dy"])
    #unpack gradient operators
    gamma0 = np.asarray(g1["gamma0"])
    gamma1 = np.asarray(g1["gamma1"])
    GRy = (gamma0/dy)
    GRx = (gamma1/dx)
    
    return GRx, GRy
    
def getGradientsConductanceGrid(gridSizeX, gridSizeY, latSpacing, lonSpacing):
    inst = instrumentParams("/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/data sources (radar and optical)/pfisr radar data/20121107.001_lp_5min-cal/20121107.001_lp_5min-cal.h5")
    
    theta = (90.0 - inst['sLat']) * np.pi/180.0 #radians. magnetic colatitude
    deltaTheta = latSpacing * np.pi/180.0
    deltaPhi = lonSpacing * np.pi/180.0 #radians
    r = 6371000 #meters. radius of Earth
    
    dx = 2.0 * r * math.sin(theta) * deltaPhi
    dy = 2.0 * r * deltaTheta    
    
    g1 = computeGradientOperators(gridSizeX, gridSizeY)
    
    gamma0 = np.asarray(g1["gamma0"])
    gamma1 = np.asarray(g1["gamma1"])
    GRy = (gamma0/dy)
    GRx = (gamma1/dx)
    
    return GRx, GRy
        
        
def computeGradientOperators(N, M):

    dx = 1
    dy = 1

    #for Gmat1-50x50_0_1.mat
    #no more little endpoint numbers!

    ftype = 1
    DerivativeOrder = 1

    MM = N * M
    minSize = 3
    inmat = np.array([-0.5, 0.0, 0.5])
    dm = 1

    A = np.zeros((MM, MM))
    for nn in range(0, MM):
        A[nn, :] = getRow_nn_transposeOper(nn, M, N)
        
        
        
    #Gamma_b takes the second derivative in y for one column, except at end points
    Gamma_b = np.zeros((M, M))
    for it in range(dm, M - dm):
        Gamma_b[it, it - dm : it + dm + 1] = inmat
        
    
    Gamma_b[dm - 1, dm - 1] = 0.0
    Gamma_b[M - dm, M - dm] = 0.0
    
    Gamma1 = np.zeros((MM, MM))
    for it in range(1, N + 1):
        Gamma1[(it - 1) * M : (it - 1) * M + M, (it - 1) * M : (it - 1) * M + M] = Gamma_b

    ##if not square, repeat with M and N reversed to form second derivative in y... but this doesn't affect gamma0 and gamma1 results, so not important for runSim....
    if M != N:
        Gamma_b = np.zeros((N, N))
        for it in range(dm, N - dm):
            Gamma_b[it, it - dm : it + dm + 1] = inmat
        Gamma_b[dm - 1, dm - 1] = 0.0
        Gamma_b[N - dm, N - dm] = 0.0
        
        Gamma2 = np.zeros((MM, MM))
        for it in range(1, M + 1):
            Gamma2[(it - 1) * N : (it - 1) * N + N, (it - 1) * N : (it - 1) * N + N] =  Gamma_b
        
        gamma = Gamma1 / (dy * dy) + np.dot(A, np.dot(Gamma2, A)) / (dx * dx)

    
    gamma0 = Gamma1 / (dy * dy)
    gamma1 = (np.dot(A, np.dot(Gamma1, A))) / (dx * dx)
    #gamma1 = np.dot(A, np.dot(Gamma2, A)) / (dx * dx)
 
    
    
    
    
    gammaOperators = {}
    gammaOperators['gamma0'] = gamma0
    gammaOperators['gamma1'] = gamma1

    return gammaOperators  
    
    
      
def getRow_nn_transposeOper(nn, M, N):
    #N is number of columns
    #M is number of columns
    
    mm = range(0, M * N)
    A = (np.ceil(nn/N) + np.fmod(nn, N) * M == mm )
    A = A.astype('d')
    return A
    
def generateGrid(N, M, sLat, sLon, mlatR, mlonR, Dlat, Dlon):
    
    #print "Building grid, size = " + str(N) + " x " + str(M)
    
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
     
    #print "Getting instrument information. Reading parameters from: " + pathToInstrumentFile 
    
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
    
    #make output dictionary
    instrument = {}
    instrument['altR'] = altR
    instrument['rngR'] = rngR
    instrument['A'] = A
    instrument['mlatR'] = mlatR
    instrument['mlonR'] = mlonR
    instrument['BabsR'] = BabsR
    instrument['Nalt'] = Nalt
    instrument['Nbeams'] = Nbeams
    instrument['sLon'] = siteMLon
    instrument['sLat'] = siteMLat
    instrument['Dlat'] = Dlat
    instrument['Dlon'] = Dlon
    instrument['time'] = time
            
    return instrument
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



def convertTimeRecordToTime(index):
    if ('dat' not in locals()) and ('var' not in locals()):
        dat = h5py.File("/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/data sources (radar and optical)/pfisr radar data/20121107.001_lp_1min-cal/20121107.001_lp_1min-cal.h5")
    time = np.asarray(dat['Time/dtime']).T
    time = np.mean(time, axis = 0)
    
    realTime = time[index]
    return realTime
    
    


def convertDecimalTimeToDictKey(time, keys):
    result = filter(lambda (lower,upper): int(lower) <= (int((time - int(time))*60)+100 * int(time)) < int(upper), zip(sorted(keys)[:-1],sorted(keys)[1:]))
    return result[0][0] if len(result) == 1 else False
def setUpMain():
    while 1:
            counter = 0
            resultsDir = "/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/run results/"
            print "Please choose a directory from one of the following options:"
            for directory in os.listdir(resultsDir):
                if counter != 0:
                    print str(directory).split('/')[-1]
                counter += 1
            
            pathToDir = raw_input()
            pathToDir = resultsDir + pathToDir
            if os.path.isdir(pathToDir):
                break
            else:
                print "Error: not a valid directory."
                
    while 1:
        lenTimeRec = raw_input("What is the length of the time records fitted for in this directory in minutes (1 or 5)? ")
        try:
            lenTimeRec = int(lenTimeRec)
            break
        except:
            print "Error: lenTimeRec must be an integer."
    
                    
    return pathToDir, lenTimeRec
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def getDictsFromExistingResults(pathToDir, oneOrFiveMin):
    counter = 0 
    
    Etotdict = {}
    Iokdict = {}
    divEdict = {}
    Imaskdict = {}
    
    readInstrument = 1
    
    for directory in os.walk(pathToDir): #loop through all measurement directories
    
        
        if counter != 0 and str(directory[0]).split('/')[-1] != "optical Plots": #ignore parent directory
            directoryPath = str(directory[0])
            directoryName = directoryPath.split('/')[-1]
            
            if oneOrFiveMin == 5:
                fileName = directoryPath +  '/' + directoryName[0:-3] + '.mat' #change to 0:-4 for 1 min
            if oneOrFiveMin == 1:
                fileName = directoryPath +  '/' + directoryName[0:-4] + '.mat'
            
            #print fileName
            
            fitResults = {}
            scipy.io.loadmat(fileName, fitResults)
            
            if readInstrument:
                X = fitResults['X']
                Y = fitResults['Y']
                mlonR = fitResults['mlonR']
                mlatR = fitResults['mlatR']
                
                pathToData = fitResults['pathToData'][0]
                
                #print pathToData
                
                inst = instrumentParams(pathToData)
                time = np.asarray(inst["time"])
                
                readInstrument = 0
                
            Etot = fitResults['Etot']
            divE = fitResults['DivE']
            Iok = fitResults['Iok']
            Imask = fitResults['Imask']
            
            if oneOrFiveMin == 5:
                timeSlice = int(directoryName[-2:]) #change to -3: for 1 min
            if oneOrFiveMin == 1:
                timeSlice = int(directoryName[-3:]) #change to -3: for 1 min
            
            
        
            decimalTime = time[timeSlice]
        
            hour = str(int(np.floor(decimalTime)))
            minute = str(int(np.round(decimalTime % 1 * 60, decimals = 0)))
            if len(minute) == 1:
                minute = "0" + minute
            clockTime = hour + minute

            Etotdict[clockTime] = Etot
            Iokdict[clockTime] = Iok
            divEdict[clockTime] = divE
            Imaskdict[clockTime] = Imask
            
            
        counter += 1 

    return X, Y, Etotdict, mlonR, mlatR, Iokdict, divEdict, Imaskdict










#load in height integrated hall and pedersen conductances for case study event
conductancesDict = {}
scipy.io.loadmat("result_AMISR_FAC_Nov07_2012.mat", conductancesDict)

#load in corresponding magnetic coordinates for the conductances
magneticCoords = {}
scipy.io.loadmat("coordconv.mat", magneticCoords)

#parse
SigmaHallTimes = conductancesDict["SigHOut"]
SigmaPallTimes = conductancesDict["SigPOut"]
mlon = magneticCoords["mlonc"][0,:]
mlat = magneticCoords["mlatc"][0,:]

mask = np.zeros((mlon.size, mlat.size))
mask[30:,:] = 1

#gradientSigmaH = np.zeros((gridSizeX, gridSizeY, 2))
#column_SigmaH = np.reshape(SigmaH[:,:,-1], (SigmaH[:,:,-1].size, 1), order = 'F')
#gradientSigmaH[:, :, 0] = np.reshape( np.dot(GRx, column_SigmaH), SigmaH[:,:,-1].shape, order = 'F')
#gradientSigmaH[:, :, 1] = np.reshape( np.dot(GRy,column_SigmaH), SigmaH[:,:,-1].shape, order = 'F')


#set up which fits to use for computing FAC's, gets fit results, define fit grid to use for interpolation of conductances
pathToDir, lenTimeRec = setUpMain()
X, Y, Etotdict, mlonR, mlatR, Iokdict, divEdict, Imaskdict = getDictsFromExistingResults(pathToDir, lenTimeRec)    

gridSizeX = SigmaHallTimes.shape[0]
gridSizeY = SigmaHallTimes.shape[1]

gridX = X[0,:]
gridY = Y[:,0]

latSpacingArray = np.zeros(mlon.size-1)
lonSpacingArray = np.zeros(mlon.size-1)
for i in range(0, mlon.size - 1):
    latSpacingArray[i] = mlat[i+1] - mlat[i]
    lonSpacingArray[i] = mlon[i+1] - mlon[i]
    
latSpacing = np.mean(latSpacingArray)
lonSpacing = np.mean(lonSpacingArray)

#do this once, then comment it out and uncomment load section below
#GRx, GRy = getGradientsConductanceGrid(gridSizeX, gridSizeY, latSpacing, lonSpacing)
#gradientsConductanceGrid = {}
#gradientsConductanceGrid["GRx"] = GRx
#gradientsConductanceGrid["GRy"] = GRy
#scipy.io.savemat('/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/gradientsConductanceGrid.mat', gradientsConductanceGrid)

#load gradient operators for conductance grid from file
gradientsConductanceGrid = {}
scipy.io.loadmat('/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/gradientsConductanceGrid.mat', gradientsConductanceGrid)
GRx = gradientsConductanceGrid["GRx"]
GRy = gradientsConductanceGrid["GRy"]

[xvalues,yvalues] = np.meshgrid(mlon, mlat)


#for each 1minute time record, interpolate conductances to fit solution grid
#use time records in range(272, 327)
for index in range(326, 327):
    
    SigmaP = SigmaPallTimes[:,:,index]
    SigmaH = SigmaHallTimes[:,:,index]
    
    #match conductance time with appropriate 5 minute fit
    conductanceTimeKey = convertTimeRecordToTime(index)
    fitTimeKey = str(convertDecimalTimeToDictKey(conductanceTimeKey, Etotdict.keys()))    
    
    Etot = Etotdict[fitTimeKey]
    divE = divEdict[fitTimeKey]
    
    Etot = np.nan_to_num(Etot)
    
    #interpolate Etot and DivE to conductance grid, ignoring nan border rows/columns
    EtotInterpolatedX = interpolate.interp2d(gridX, gridY, Etot[:, :, 0], kind = 'cubic')
    EtotInterpolatedY = interpolate.interp2d(gridX, gridY, Etot[:, :, 1], kind = 'cubic')
    divEInterpolated = interpolate.interp2d(gridX[2:-2], gridY[2:-2], divE[2:-2, 2:-2], kind = 'cubic')
    
    EtotCondGrid = np.zeros((mlon.size, mlat.size, 2))
    EtotCondGrid[:, :, 0] = EtotInterpolatedX(mlon, mlat) * mask
    EtotCondGrid[:, :, 1] = EtotInterpolatedY(mlon, mlat) * mask
    divECondGrid = divEInterpolated(mlon, mlat)
    
    divInterpolatedEtotCondGrid = np.reshape(np.dot(GRx, np.reshape(EtotCondGrid[:,:,0], (GRx.shape[0], 1), order = 'F'))  + np.dot(GRy, np.reshape(EtotCondGrid[:,:,1], (GRx.shape[0], 1), order = 'F')), EtotCondGrid[:,:,0].shape, order = 'F')
#    plt.figure(figsize = (12,10.5))
 #   plt.pcolormesh(mlon, mlat, divECondGrid * mask, cmap = 'jet', vmin = np.nanmin(divE), vmax = np.nanmax(divE))
 #   plt.colorbar(cmap = 'jet')
    
 #   plt.figure(figsize = (12,10.5))
 #   plt.pcolormesh(mlon, mlat, divInterpolatedEtotCondGrid * mask, cmap = 'jet', vmin = np.nanmin(divE), vmax = np.nanmax(divE))
 #   plt.colorbar(cmap = 'jet')
    
    #plt.show()
    
    #########check interpolations by plotting######
    #plt.figure(figsize = (12,10.5))
    #plt.pcolormesh(mlon, mlat, divECondGrid, cmap = 'jet', vmin = np.nanmin(divE), vmax = np.nanmax(divE))
    #plt.colorbar(cmap = 'jet')
    #plt.figure(figsize = (12, 10.5))
    #plt.pcolormesh(gridX, gridY, divE, cmap = 'jet', vmin = np.nanmin(divE), vmax = np.nanmax(divE))
    #plt.axis([mlon.min(), mlon.max(), mlat.min(), mlat.max()])
    #plt.colorbar(cmap = 'jet')
    #mlon = np.reshape(mlon, (1, mlon.size))
    #mlat = np.reshape(mlat, (1, mlat.size))
    #[mlonX,mlatY] = np.meshgrid(mlon, mlat)
    #plt.figure(figsize = (12,12))
    #Q1 = quiver( mlonX, mlatY, EtotCondGrid[:, :, 0], EtotCondGrid[:, :, 1], color = 'r', width = 0.003)
    #plt.figure(figsize = (12, 12))
    #Q1 = quiver( X, Y, Etot[:, :, 0], Etot[:, :, 1], color = 'r', width = 0.003)
    #plt.axis([mlon.min(), mlon.max(), mlat.min(), mlat.max()])
    #plt.show()
    
    numGridEntries = mlon.size * mlat.size
    
    ####term1: SigmaP ( divergence of E)
    term1 = SigmaP * divInterpolatedEtotCondGrid
    
    ####term2: gradSigmaP dot E
    #procedure: reshape SigmaPfitGrid and E into column vectors. then perform this calculation: d/dx SigmaP dot Ex + d/dy SigmaP dot Ey. then reshape back into grid.
    column_SigmaPCondGrid = np.reshape(SigmaP, (SigmaP.size, 1), order = 'F')
    column_Ex = np.reshape(EtotCondGrid[:, :, 0], (numGridEntries, 1), order = 'F')
    column_Ey = np.reshape(EtotCondGrid[:, :, 1], (numGridEntries, 1), order = 'F')
    column_term2 = np.dot(GRx, column_SigmaPCondGrid) * column_Ex + np.dot(GRy, column_SigmaPCondGrid) * column_Ey
    term2 = np.reshape(column_term2, SigmaP.shape, order = 'F')
    
    ####term3:  gradSigmaH dot (E cross Bhat) 
    #procedure: we've defined our coordinate system so that Bhat at the north pole is aligned with -z hat. Therefore E cross Bhat = Ex yhat - Ey xhat. dotting that into gradSigmaH, we get -d/dx SigmaH Ey + d/dy SigmaH Ex. once again, transform everything into column vectors, then take the derivatives, then put back on the grid.
    column_SigmaHCondGrid = np.reshape(SigmaH, (SigmaH.size, 1), order = 'F')
    column_term3 = -np.dot(GRx, column_SigmaHCondGrid) * column_Ey + np.dot(GRy, column_SigmaHCondGrid) * column_Ex
    term3 = np.reshape(column_term3, SigmaH.shape, order = 'F')
    
    Jparallel = (term1 + term2 - term3) * mask
    
    
    
    
    
    
    
    
    
    #checking Roger's identities:
    #1: div(SigP E ) = SigP divE + E dot grad SigP
    #LHS = np.reshape(np.dot(GRx, np.reshape(SigmaP * EtotCondGrid[:,:,0], (SigmaP.size, 1), order = 'F')) +  np.dot(GRy, np.reshape(SigmaP * EtotCondGrid[:,:,1], (SigmaP.size, 1), order = 'F')), SigmaP.shape, order = 'F' )
    
    #RHS = SigmaP * np.reshape(np.dot(GRx, np.reshape(EtotCondGrid[:,:,0], (SigmaP.size, 1), order = 'F')) +  np.dot(GRy, np.reshape(EtotCondGrid[:,:,1], (SigmaP.size, 1), order = 'F')), SigmaP.shape, order = 'F' ) + EtotCondGrid[:,:,0] *  np.reshape(np.dot(GRx, np.reshape(SigmaP, (SigmaP.size, 1), order = 'F')), SigmaP.shape, order = 'F' ) + EtotCondGrid[:,:,1] *  np.reshape(np.dot(GRy, np.reshape(SigmaP, (SigmaP.size, 1), order = 'F')), SigmaP.shape, order = 'F' ) 
    
    
    #2: div(E cross Bhat) = 0
    #LHS = np.reshape( np.dot(GRx, np.reshape(-EtotCondGrid[:,:,1], (SigmaP.size, 1), order = 'F')) + np.dot(GRy, np.reshape(EtotCondGrid[:,:,0], (SigmaP.size, 1), order = 'F'))     , SigmaP.shape, order = 'F')
    
    
    #3: 
    #LHS = (np.reshape( np.dot(GRx, np.reshape(-SigmaH * EtotCondGrid[:,:,1], (SigmaP.size, 1), order = 'F')) + np.dot(GRy, np.reshape(SigmaH * EtotCondGrid[:,:,0], (SigmaP.size, 1), order = 'F'))     , SigmaP.shape, order = 'F')) * mask
    
    #RHS =( EtotCondGrid[:,:,0] *  np.reshape(np.dot(GRy, np.reshape(SigmaH, (SigmaP.size, 1), order = 'F')), SigmaP.shape, order = 'F' ) - EtotCondGrid[:,:,1] *  np.reshape(np.dot(GRx, np.reshape(SigmaH, (SigmaP.size, 1), order = 'F')), SigmaP.shape, order = 'F' ) ) * mask
    
    
    #plt.figure(figsize = (15, 5))
    #ax1 = subplot(1,3,1)
    #plt.contour(mlon, mlat, LHS , cmap = 'jet', vmin = np.nanmin(LHS), vmax = np.nanmax(LHS) , aspect = 'auto', linewidths = 3.0 )
    #plt.title('Equivalence 1: LHS = 0')   
    #plt.colorbar(format='%.0e')
    
    #ax2 = subplot(1,3,2)
    #plt.contour(mlon, mlat, RHS , cmap = 'jet', vmin = np.nanmin(LHS), vmax = np.nanmax(LHS) , aspect = 'auto', linewidths = 3.0 )    
    #plt.title('Equivalence 1: RHS')   
    #plt.colorbar(format='%.0e')
    
    #ax3 = subplot(1,3,3)
    #plt.contour(mlon, mlat, LHS - RHS , cmap = 'jet' , aspect = 'auto', linewidths = 3.0 ) 
    #plt.title('Equivalence 1: LHS - RHS')   
    #plt.colorbar(format='%.0e')
    
    #plt.show()
    #sys.exit()
    
    
    #E cross Bhat = Ex yhat - Ey xhat
    
    
    
    
    
    
    
    
    
    
    
    
    
    #picture time! plot FAC 
    Jmax = np.nanmax(Jparallel)* 1000000.0
    Jmin = np.nanmin(Jparallel)* 1000000.0
    orig_cmap = matplotlib.cm.seismic
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(Jmax)/(np.absolute(Jmax) + np.absolute(Jmin))), name = 'shifted')
    
    plt.figure(figsize = (5, 4))
    print mlon.shape
    print mlat.shape
    print Jparallel.shape
    plt.contour(mlon, mlat, Jparallel * 1000000.0, cmap = shifted_cmap, vmin = np.nanmin(Jparallel * 1000000.0), vmax = np.nanmax(Jparallel * 1000000.0))
    plt.colorbar()
    plt.title('Original Current Formula')
  
    
    
    
    
    
    
    
    term1Horizontal = np.zeros(EtotCondGrid.shape)
    term1Horizontal[:,:,0] = SigmaP * EtotCondGrid[:,:,0]
    term1Horizontal[:,:,1] = SigmaP * EtotCondGrid[:,:,1]
    
    term2Horizontal = np.zeros(EtotCondGrid.shape)
    term2Horizontal[:,:,0] = -SigmaH * -EtotCondGrid[:,:,1]
    term2Horizontal[:,:,1] = -SigmaH * EtotCondGrid[:,:,0]
    
    currentDensity = np.zeros(EtotCondGrid.shape)
    currentDensity = term1Horizontal + term2Horizontal
    
    JparMethodTwo = np.zeros(Jparallel.shape)
    JparMethodTwo = np.reshape(np.dot(GRx, np.reshape(currentDensity[:,:,0], (currentDensity[:,:,0].size, 1), order = 'F')) + np.dot(GRy, np.reshape(currentDensity[:,:,1], (currentDensity[:,:,1].size, 1), order = 'F')), JparMethodTwo.shape, order = 'F')
    
    Jmax = np.nanmax(JparMethodTwo)* 1000000.0
    Jmin = np.nanmin(JparMethodTwo)* 1000000.0
    orig_cmap = matplotlib.cm.seismic
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(Jmax)/(np.absolute(Jmax) + np.absolute(Jmin))), name = 'shifted')
    plt.figure(figsize = (5,4))
    plt.contour(mlon, mlat, JparMethodTwo* 1000000.0 * mask, cmap = shifted_cmap, vmin = np.nanmin(JparMethodTwo * 1000000.0), vmax = np.nanmax(JparMethodTwo * 1000000.0))
    plt.colorbar()
    plt.title('Horizontal Current Formula')
    
    #plt.show()
    #sys.exit()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    JparDict = {}
    JparDict["J"] = Jparallel
    JparDict["sigP_term1"] = term1
    JparDict["gradSigP_term2"] = term2
    JparDict["minus_gradSigH_term3"] = -term3
    JparDict["mlon"] = mlon
    JparDict["mlat"] = mlat
    scipy.io.savemat('/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/Jpar_withterms1204.mat', JparDict)
    
    
    
    
 #   plt.figure(figsize = (5, 4))
 #   plt.pcolormesh(mlon, mlat, Jparallel, cmap = shifted_cmap, vmin = Jmin, vmax = Jmax)
 #   plt.colorbar(format = '%.0e')
    
 #   plt.figure(figsize = (5, 4))
 #   plt.pcolormesh(mlon, mlat, term1 * mask, cmap = shifted_cmap, vmin = Jmin, vmax = Jmax)
 #   plt.colorbar(format = '%.0e')
    
 #   plt.figure(figsize = (5, 4))
 #   plt.pcolormesh(mlon, mlat, term2 * mask, cmap = shifted_cmap, vmin = Jmin, vmax = Jmax)
 #   plt.colorbar(format = '%.0e')
    
 #   plt.figure(figsize = (5, 4))
 #   plt.pcolormesh(mlon, mlat, term3 * mask, cmap = shifted_cmap, vmin = Jmin, vmax = Jmax)
 #   plt.colorbar(format = '%.0e')
    
 #   plt.show()
    
    #save figure. last term converts decimal time to clock time. i.e. 8.154 --> 8:09
    #figureName = pathToDir + "/FAC_condGrid_" + str(int((conductanceTimeKey - int(conductanceTimeKey))*60)+100 * int(conductanceTimeKey)) + ".jpg"
    #savefig(figureName)
    
    plt.show()

    
    
