from constants import *
from lostInTranslation import *
import os
import shutil
import scipy.io
from numpy import *
from simPattern import *
from scipy import interpolate, spatial, stats
from generateGrid import *
from gradientsAndGMatrix import *
import h5py


#########################################################################################################
#########################################################################################################
#########################################################################################################



def simulateData(Perr, label, altRshape, I, Igood, x, y, dx, dy, X, Y, A, Nalt, Nbeams, mlonR, mlatR, rngR, Dlon, Dlat, IRANDOR = 1):
    
    simName = "sim_" + label 
    
    randormat = scipy.io.loadmat(os.path.abspath('randor.mat')) 
    randor = np.asarray(randormat["randor"]) 
    
    #simPattern calculates simulated potential, E-field, div E, and true vector velocities, and returns divETrue, EtotTrue, vVecTrue
    divETrue, EtotTrue, vVecTrue = simPattern(x, y, X, Y, dx, dy, Dlon, Dlat) 
        
    dvlosR = np.zeros(altRshape) 
        
    dvlosR[I,:] = NaN 
    
    Xspline = X[0, 0:].T
    Yspline = Y[0:, 0]
    v1 = np.full((Nalt * Nbeams, 2), NaN) 
    

    vspline = interpolate.RectBivariateSpline(Yspline, Xspline, vVecTrue[:,:,0], kx = 1, ky = 1)
    v1[Igood, 0] = vspline.ev( mlatR[Igood][0:, 0], mlonR[Igood][0:, 0]) 
   
       
    vspline = interpolate.RectBivariateSpline(Yspline, Xspline, vVecTrue[:,:,1], kx = 1, ky = 1) 
    v1[Igood, 1] = vspline.ev( mlatR[Igood][0:, 0], mlonR[Igood][0:, 0]) 
        
                
    Adot = np.asmatrix(A[:,0:2]).H 
    vdot = np.asmatrix(v1).H 
    
    trueVlosR = MATLABdot(Adot, vdot).T 
    
    dvlosR = np.polyval(Perr, rngR) # + np.random.standard_normal((rngR.shape)) / 10.0 
    
    tv = np.zeros(trueVlosR.shape)    
    tv[:, 0] = trueVlosR[:, 0] + dvlosR[:, 0] * randor[0:(Nalt * Nbeams), IRANDOR - 1] 
        
    tdv = np.zeros(dvlosR.shape) 
    tdv[:, 0] = dvlosR[:,0]     
    
    return simName, tv, tdv, trueVlosR, dvlosR, divETrue, EtotTrue, vVecTrue, randor
    
    
#########################################################################################################
#########################################################################################################
#########################################################################################################

    
    
def importRealData(pathToData, timeSlice, label = ""):
    
    if label != "":
        simName = "real_" + label
    else:   
        simName = "real_" + pathToData.split("/")[1] + "_" + pathToData.split("/")[2]
    
    dat = h5py.File(pathToData)

    fits = np.asarray(dat['FittedParams/Fits']).T
    errors = np.asarray(dat['FittedParams/Errors']).T
      
    fixTimeSpeciesVlosData = fits[-1, 0, :, :, timeSlice ]
    errorsFixTimeSpeciesVlosData = errors[-1, 0, :, :, timeSlice]
        
    tv = np.reshape(fixTimeSpeciesVlosData, (fixTimeSpeciesVlosData.size, 1), order = 'F')
    tdv = np.reshape(errorsFixTimeSpeciesVlosData, (errorsFixTimeSpeciesVlosData.size, 1), order = 'F')
      
    #RHV add access to SNR
    snr_full=np.asarray(dat['NeFromPower/SNR'])
    ranges_full=np.asarray(dat['NeFromPower/Range'])
  
    ranges=np.asarray(dat['FittedParams/Range'])
    
    snr=np.zeros(fixTimeSpeciesVlosData.shape)
    for b in range(snr_full.shape[1]):
        snr[:,b]=interpolate.interp1d(ranges_full[0,:],snr_full[timeSlice,b,:])(ranges[b,:])
            
    tsnr = np.reshape(snr, (snr.size,1), order = 'F')

    print 'shapes', tv.shape,tdv.shape,tsnr.shape

    dat.close()
    
    return simName, tv, tdv, tsnr
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
