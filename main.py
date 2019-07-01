from runSim import *
from setUp import *
from lostInTranslation import getMask, ind2sub2
from PIL import Image, ImageDraw
from matplotlib.path import Path
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from hardCodeOverride import *
from getIokUnion import *
import sys

#sys.path.append(os.path.join(data_dir,'opticalPlotter.py'))
#import opticalPlotter 

# RHV opticalPlotter requires pyfits to read fits image files

def setUpMain():
    #RHV this is unnecessary...
    while 1:
        #getFromExistingResults = raw_input("Would you like to run the fitter (0) or load results from an existing file (1)? ")
        getFromExistingResults=0
        try:
            getFromExistingResults = int(getFromExistingResults)
            break
        except:
            print "Error: getFromExistingResults must be an integer."
            
    if getFromExistingResults:
        pass #RHV
        '''
        while 1:
            counter = 0
            #resultsDir = "/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/run results/"
            resultsDir = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/"
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
        '''
    else:
        pathToDir = None
        lenTimeRec = None
                    
    return getFromExistingResults, pathToDir, lenTimeRec

def runFitter(config_file,experiment_dir):
#RHV removed useIokIntersection as an input. Now comes from ini file.

    #### settings:
    # use1d: turn on to use 1d solution in final result.
    # use2d: turn on to use 2d solution in final result.
    # doChiSquare: turn on to calculate 1d solution goodness of fit and decide whether or not to compute the 2d solution. Good for 1d potentials. 

    use1d = 0
    use2d = 1
    doChiSquare = 0
    doVparallel = 1
    # gridSizeX = 23
    # gridSizeY = 10
    gridSizeX = 50
    gridSizeY = 50

    hardCodeOverrideFlag = 0

    if hardCodeOverrideFlag:
        print "Hard code override flag is turned on. Algorithm execution parameters read in from hardCodeOverride.py."
        numProcesses, simFlag, realFlag, derivativeWeightDict, hardCode = hardCodeOverride()
        masksc = []
        pathsToData = []
        label = []
        Perr = []
        for i in range(len(simFlag)):
            masksc.append(hardCode[i]["masksc"])
            pathsToData.append(hardCode[i]["pathToData"])
            label.append(hardCode[i]["label"])
            Perr.append(hardCode[i]["Perr"])
   
    else:
        #numProcesses, simFlag, realFlag, Perr, label, masksc, derivativeWeightDict, pathsToData = setUp()
        opts = setUpini(config_file,experiment_dir) #RHV sys.argv[1] should be the name of an ini file
        numProcesses=opts['numProcesses']
        simFlag=opts['simFlag']
        realFlag=opts['realFlag']
        Perr=opts['Perr']
        label=opts['label']
        masksc=opts['masksc']
        derivativeWeightDict=opts['derivativeWeightDict']
        pathsToData=opts['pathsToData']
        recStart=opts['recStart']
        recStop=opts['recStop']
        useIokIntersection=opts['useIokIntersection']
        
        hardCode = [0 for i in range(numProcesses)]
        


    #multiprocessing pipeline, fix later.
    for i in range(0, numProcesses):
        
        if useIokIntersection:
            #get intersection of Ioks
            IokIntersection = getIokUnion(hardCodeOverrideFlag, simFlag[i], realFlag[i], masksc[i], derivativeWeightDict[i], gridSizeX, gridSizeY, hardCode[i], pathsToData[i], label[i], Perr[i], use1d, use2d, doChiSquare, doVparallel, recStart=recStart, recStop=recStop)
        else:
            IokIntersection = 0
        
        print "\n\n~~~~~~~~~~~~~~~~~~~~~~ Running Process #" + str(i + 1) + ": ~~~~~~~~~~~~~~~~~~~~~~"
        if simFlag[i]:
            print "Simulation. Error style = " + str(Perr[i]) + ". Mask style = " +  str(masksc[i]).replace('\n', '')
        else:
            print "Real Data. Source: " + pathsToData[i] + ". Mask style = " +  str(masksc[i]).replace('\n', '') + "\n "
        
        
        X, Y, Etot, mlonR, mlatR, Iok, divEdict, Imaskdict, labeldict = runSim(hardCodeOverrideFlag, simFlag[i], realFlag[i], masksc[i], derivativeWeightDict[i], gridSizeX, gridSizeY, hardCode[i], pathsToData[i], label[i], Perr[i], use1d, use2d, doChiSquare, doVparallel, IokInput = IokIntersection, recStart=recStart, recStop=recStop)
    
        #mp.Process(target = runSim, args = (hardCodeOverrideFlag, simFlag[i], realFlag[i], masksc[i], derivativeWeightDict[i], gridSizeX, gridSizeY, hardCode[i], pathsToData[i], use1d, use2d, doChiSquare, doVparallel)).start()   
        
    
    return X, Y, Etot, mlonR, mlatR, np.asarray(Iok), divEdict, Imaskdict, labeldict

    

#convertToInterval() takes a time stamp corresponding to a single allSky picture and finds the corresponding five minute fit. 
def convertToInterval(time, intervals):
    sortedIntervals = sorted(map(lambda a: int(a), intervals))
    return filter(lambda (a,b): a <= int(time / 10) < b,zip(sortedIntervals[:-1], sortedIntervals[1:]))[0][0]
    
def in_hull(p, hull): ##CODE FROM STACKOVERFLOW
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
def getDictsFromExistingResults(pathToDir, oneOrFiveMin):
    counter = 0 
    
    Etotdict = {}
    Iokdict = {}
    divEdict = {}
    Imaskdict = {}
    
    for directory in os.walk(pathToDir): #loop through all measurement directories
    
        
        if counter != 0 : #ignore parent directory
            directoryPath = str(directory[0])
            directoryName = directoryPath.split('/')[-1]
            
            if oneOrFiveMin == 5:
                fileName = directoryPath +  '/' + directoryName[0:-3] + '.mat' 
            if oneOrFiveMin == 1:
                fileName = directoryPath +  '/' + directoryName[0:-4] + '.mat'
            
            print fileName
            
            fitResults = {}
            scipy.io.loadmat(fileName, fitResults)
            
            if counter == 1:
                X = fitResults['X']
                Y = fitResults['Y']
                mlonR = fitResults['mlonR']
                mlatR = fitResults['mlatR']
                
                pathToData = fitResults['pathToData'][0]
                
                print pathToData
                
                inst = instrumentParams(pathToData)
                time = np.asarray(inst["time"])
                
            Etot = fitResults['Etot']
            divE = fitResults['DivE']
            Iok = fitResults['Iok']
            Imask = fitResults['Imask']
            
            if oneOrFiveMin == 5:
                timeSlice = int(directoryName[-2:]) 
            if oneOrFiveMin == 1:
                timeSlice = int(directoryName[-3:]) 
            
            
        
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



if __name__ == "__main__":


    getFromExistingResults, pathToDir, lenTimeRec = setUpMain()
    
    '''
    # az cal file/ elevation cal file
    az_cal = os.path.join(data_dir,'PKR_DASC_20110112_AZ_10deg.FITS')
    el_cal = os.path.join(data_dir,'PKR_DASC_20110112_EL_10deg.FITS')
    
    #DIRECTORY CONTAINED OPTICAL IMAGES
    in_dir = os.path.join(data_dir, '20121107') ###SET THIS TO BE THE DIRECTORY WITH THE ALLSKY IMAGES YOU WISH TO PLOT
    '''
    
    if getFromExistingResults:
        out_dir = os.path.join(pathToDir, "optical Plots")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        X, Y, Etotdict, mlonR, mlatR, Iokdict, divEdict, Imaskdict = getDictsFromExistingResults(pathToDir, lenTimeRec)
    else:
        ##get dicts of fits. X Y, mlonR, mlatR are for grid. Etotdict and Iokdict are results. keys to dicts are time slices in string format, ie 850 = 8:50. 
        
        ###IMPORTANT: SET USEIOKINTERSECTION BELOW TO USE/DISABLE IOK INTERSECTION SET
    
        #RHV: useIokIntersection has now been moved into the config file. 
        #Default should be 0. I do not recommend using 1 on long experiments
        X, Y, Etotdict, mlonR, mlatR, Iokdict, divEdict, Imaskdict, label = runFitter()
        #this is only the out_dir for the (disabled) optical plots
    #    out_dir = os.path.join("/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/",label.values()[0],"optical Plots")
    
    #os.mkdir(out_dir)
    
    
    
    #del divEdict[str(max([int(x) for x in divEdict.keys()]))]        
    #divEGlobalMax = np.nanmax(np.nanmax(np.nanmax(divEdict.values())))
    #divEGlobalMin = np.nanmin(np.nanmin(np.nanmin(divEdict.values())))
    
    
    '''
    #Remove optical plotting for now....
    counter = 0
    for i in os.listdir(in_dir): 
        if counter != 0: #ignore parent directory
            input = os.path.join(in_dir,i)  
               
            #this ugly but short line of code takes the optical data file name and parses it to pick out the clock time of the image, so that it can be compared to the fit result clocktime. 
            opticalPictureTime = float(i.split('.')[0].split('_')[-1][1:5])
            
            try: 
                opticalTimeKey = str(convertToInterval(opticalPictureTime, Etotdict.keys()))
                
                #find radar mask so that we only pass the fits within the mask to the optical plotter
                Iok = Iokdict[opticalTimeKey] 
                points = np.vstack((mlonR[Iok], mlatR[Iok]))[:,:,0].T
                hull = ConvexHull(points)
                hull_path = Path( points[hull.vertices] )
                xCoords = X[0,:]
                yCoords = Y[:,0]
                newMask = np.zeros((xCoords.size, yCoords.size))
                for i in range(0,xCoords.size):
                    for j in range(0,yCoords.size):
                        newMask[j,i] = hull_path.contains_point((xCoords[i], yCoords[j]))   
                Emasked = np.zeros(Etotdict[opticalTimeKey].shape)
                divEmasked = np.zeros(divEdict[opticalTimeKey].shape)
                Emasked[:,:,0] = Etotdict[opticalTimeKey][:,:,0] * newMask
                Emasked[:,:,1] = Etotdict[opticalTimeKey][:,:,1] * newMask
                divEmasked = divEdict[opticalTimeKey] * newMask
                Emasked[Emasked==0.0]=numpy.nan
                
            
                print opticalTimeKey
                
                #plot all the things!
                opticalPlotter.plot_allsky_FA(input, out_dir, az_cal, el_cal, X, Y, mlonR, mlatR, Emasked, Iokdict[opticalTimeKey], divEdict[opticalTimeKey], Imaskdict[opticalTimeKey], divEGlobalMax, divEGlobalMin, opticalTimeKey)         
            except IOError:
                print str(i) + " is corrupted."
            except KeyError:
                print "Key Error: No fit exists for time " + str(opticalPictureTime)
            except IndexError:
                print "Index Error: No fit exists for time " + str(opticalPictureTime)
    
        counter += 1
    '''
        
    
    
    
    
    
    
    
    

    
    
    
    
