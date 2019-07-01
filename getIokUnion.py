from constants import *
from lostInTranslation import *
import os
import shutil
import scipy.io
import numpy as np
from simPattern import *
from scipy import interpolate, spatial, stats
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from gradientsAndGMatrix import *
from resolveEfield import *
import numpy.matlib
from plotProcess import *
from generateGrid import *
from setUp import *
from getData import *

def getIokUnion(hardCodeOverride, simFlag, realFlag, masksc, derivWeights, gridSizeX, gridSizeY, hardCode, pathToData = None, label = None, Perr = None, use1d = 1, use2d = 1, doChiSquare = 1, dovpar = 1, IokInput = None, recStart=0, recStop=10):
#RHV added recStart and recStop 
     
    if hardCodeOverride:
        Perr = hardCode["Perr"]
        label = hardCode["label"]
        pathToData = hardCode["pathToData"]
    
    #########################################################################################################
    ######################### get grid, instrument parameters, gradient operators ###########################
    #########################################################################################################
    
    #generate grid, get instrument parameters, make gradient operators. functions return dictionaries
    inst = instrumentParams(pathToData)
        
    #unpack instrument parameters
    BabsR = inst["BabsR"]/inst["BabsR"] * BabsFixed 
    mlonR = np.asarray(inst["mlonR"])
    mlatR = np.asarray(inst["mlatR"])
    Dlat = np.asarray(inst["Dlat"])
    Dlon = np.asarray(inst["Dlon"])
    Nalt = np.asarray(inst["Nalt"], dtype = np.int32)
    Nbeams = np.asarray(inst["Nbeams"], dtype = np.int32)
    rngR = np.asarray(inst["rngR"])
    A = np.asarray(inst["A"]) 
    ElR = np.asarray(inst["ElR"], dtype = np.float64)
    altR = np.asarray(inst["altR"])    
    with np.errstate(invalid='ignore'):
        I =  np.where(((altR < minAlt) | (altR > maxAlt)))[0] 
        
    
    
    
    Ae = np.zeros(A.shape) 
    A[I,:] = NaN 
    Ae[:, 0] = A[:, 1] 
    Ae[:, 1] = -A[:, 0]  ##cross product
    Ae[:, 2] = A[:, 2] 
    
       
    Igood = np.where(((~np.isnan(mlonR)) & (~np.isnan(mlatR))))[0]
    Iokdict = {}
    
    #RHV added SNR
    #for timeSlice in range(54, 66):
    for timeSlice in range(recStart,recStop): #RHV: sloppy, must be changed both here and in runSim   
        print "Importing real data from " + pathToData + ", timeSlice # " + str(timeSlice)
        if hardCodeOverride:
            simName, tv, tdv, tsnr = importRealData(pathToData, timeSlice, label)
        else:
            simName, tv, tdv, tsnr = importRealData(pathToData, timeSlice)


        #choose good measurements
        tv = np.reshape(tv, tv.size, order = 'F')
        tdv = np.reshape(tdv, tdv.size, order = 'F')
        tsnr = np.reshape(tsnr, tsnr.size, order = 'F')
        ElR = np.reshape(ElR, ElR.size, order = 'F')

        AeCol1 = Ae[:, 0]
        AeCol2 = Ae[:, 1]
        AeCol3 = Ae[:, 2]


        #RHV added SNR
        with np.errstate(invalid='ignore'):       
            Iok = np.where(( (np.isfinite(AeCol1)) & (np.isfinite(AeCol2)) & (np.isfinite(AeCol3)) &       (np.isfinite(tv)) & (np.isfinite(tdv)) & (np.less(tdv, maxDv)) & (np.less(np.absolute(tv), maxV)) & (np.less(ElR, maxEl)) & (np.greater(tsnr,minSNR))))[0]
        Iokdict[timeSlice] = Iok

        #RHV sanity check
        '''
        for i in range(tdv.shape[0]):
            print i,AeCol1[i],AeCol2[i],AeCol3[i],tv[i],tdv[i],ElR[i],i in Iok
        '''

    IokIntersection = reduce(lambda a,b: filter(lambda c: c in a,b), Iokdict.values())
    return IokIntersection
            
            
            
            
        
