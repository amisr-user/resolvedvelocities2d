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

# np.set_printoptions(threshold=np.nan)

def runSim(hardCodeOverride, simFlag, realFlag, masksc, derivWeights, gridSizeX, gridSizeY, hardCode, pathToData = None, label = None, Perr = None, use1d = 0, use2d = 1, doChiSquare = 1, dovpar = 1, IokInput = 0, recStart=0, recStop=10): 
    if hardCodeOverride:
        Perr = hardCode["Perr"]
        label = hardCode["label"]
        pathToData = hardCode["pathToData"]
    
    #########################################################################################################
    ######################### get grid, instrument parameters, gradient operators ###########################
    #########################################################################################################
    
    #generate grid, get instrument parameters, make gradient operators. functions return dictionaries
    
    print pathToData
    
    
    inst = instrumentParams(pathToData)
        
    #unpack instrument parameters
    BabsR = inst["BabsR"]/inst["BabsR"] * BabsFixed ##NEED FOR BOTH
    mlonR = np.asarray(inst["mlonR"])
    mlatR = np.asarray(inst["mlatR"])
    Dlat = np.asarray(inst["Dlat"])
    Dlon = np.asarray(inst["Dlon"])
    Nalt = np.asarray(inst["Nalt"], dtype = np.int32) 
    Nbeams = np.asarray(inst["Nbeams"], dtype = np.int32)
    rngR = np.asarray(inst["rngR"])
    A = np.asarray(inst["A"]) ##NEED FOR BOTH
    ElR = np.asarray(inst["ElR"], dtype = np.float64)
    altR = np.asarray(inst["altR"])    
    time = np.asarray(inst["time"])
    with np.errstate(invalid='ignore'):
        I =  np.where(((altR < minAlt) | (altR > maxAlt)))[0] ##NEED FOR BOTH
        
        
    grid = generateGrid(gridSizeX, gridSizeY, inst["sLat"], inst["sLon"], mlatR, mlonR, Dlat, Dlon)
    
    
    g1 = getGradientOperators(grid['Nx'], grid['Ny'])
        
    #unpack grid results
    x = np.asarray(grid["x"])
    y = np.asarray(grid["y"])
    X = np.asarray(grid["X"])
    Y = np.asarray(grid["Y"])
    dx = np.asarray(grid["dx"])
    dy = np.asarray(grid["dy"])
    dlat = double(grid["dlat"])
    dlon = double(grid["dlon"])
    Nphi = np.asarray(grid["Nphi"])
    XR = np.asarray(grid["XR"])
    YR = np.asarray(grid["YR"])
    
    
    
    
    
    
    #unpack gradient operators
    gamma0 = np.asarray(g1["gamma0"])
    gamma1 = np.asarray(g1["gamma1"])
    GRy = (gamma0/dy)
    GRx = (gamma1/dx)
    
    
    Ae = np.zeros(A.shape) ##NEED FOR BOTH
    A[I,:] = NaN ##NEED FOR BOTH
    Ae[:, 0] = A[:, 1] ##NEED FOR BOTH
    Ae[:, 1] = -A[:, 0] ##NEED FOR BOTH ##cross product
    Ae[:, 2] = A[:, 2] ##NEED FOR BOTH
    
       
    Igood = np.where(((~np.isnan(mlonR)) & (~np.isnan(mlatR))))[0]
    
    
    Etotdict = {}
    Iokdict = {}
    divEdict = {}
    Imaskdict = {}
    labeldict = {}
    
    makeNewDirectory = 1
    
    #for 11.07.12 pfisr arc event from 8:30 to 9:06ish
   # for timeSlice in range(271, 330): #1 minute 271 - 330
    #for timeSlice in range(54, 66): #five minute 54 - 66
    for timeSlice in range(recStart,recStop): #RHV: sloppy, must be changed both here and in runSim
        #########################################################################################################
        ######################################## SIMULATE OR IMPORT DATA ########################################
        #########################################################################################################
    
        #get data! either simulate it or import from real measurements. 
        if simFlag:
            print "Simulating potential pattern and data with error style " + label + "." 
            simName, tv, tdv, trueVlosR, dvlosR, divETrue, EtotTrue, vVecTrue, randor = simulateData(Perr, label, altR.shape, I, Igood, x, y, dx, dy, X, Y, A, Nalt, Nbeams, mlonR, mlatR, rngR, Dlon, Dlat)
        
        elif realFlag:
            #RHV added snr
            print "Importing real data from " + pathToData + ", timeSlice # " + str(timeSlice)
            if hardCodeOverride:
                simName, tv, tdv, tsnr = importRealData(pathToData, timeSlice, label)
            else:
                simName, tv, tdv, tsnr = importRealData(pathToData, timeSlice)
         
         
    
        
        #########################################################################################################
        ########################################## Set up directories ###########################################
        #########################################################################################################

        #RHV this needs to be cleaned up

        # set up directories
        simName = label + "_" + str(timeSlice)
        simPath = os.getcwd() + "/run_results/" + simName 
        
        directoryName = simPath.split('/')[-1].split('_')
        directoryName.pop()
        directoryName = '_'.join(directoryName)
        
        directoryPath = os.getcwd() + "/run_results/" + directoryName
        simPath = directoryPath + "/" + simName
        
        
        if makeNewDirectory and os.path.isdir(directoryPath):
            shutil.rmtree(directoryPath)
        if makeNewDirectory:
            os.mkdir(directoryPath)
            makeNewDirectory = 0
        if os.path.isdir(simPath):
            shutil.rmtree(simPath)
        
        os.mkdir(simPath)
   
   
        #########################################################################################################
        ############################################ Run calculations ###########################################
        #########################################################################################################
    
    
        #for 1D solution
        yy = np.array([-dlat, 0.0, dlat])
        zz = np.array([0.0, 1.0, 0.0])
    
        Nsol = int(y.size * 2.0)
        fm0 = np.zeros((Nalt * Nbeams, Nsol))
        a = np.full(y.size, NaN)
            
        for i in range(0, Nalt * Nbeams):
            if ~np.isnan(mlatR[i]):
                #ai = interpolate.InterpolatedUnivariateSpline(mlatR[i] + yy, zz, k=1, ext = 1 )
                ai = interpolate.InterpolatedUnivariateSpline(mlatR[i] + yy, zz, k=1)
                a = ai.__call__(y[0,0:])
                iia = np.nonzero(a)[0]
                        
                fm0[i, iia] = Ae[i, 0] * a[iia]
                fm0[i, y.size + iia] = Ae[i, 1] * a[iia]       
                      
    
        #for 2D solution

        ttx = np.zeros((Nalt * Nbeams, Nphi))
        tty = np.zeros((Nalt * Nbeams, Nphi))

        [xx, yy] = np.meshgrid([-dlon, 0.0, dlon], [-dlat, 0.0, dlat])
        zz = np.zeros((3,3))
        zz[1, 1] = 1.0
        a = np.full(XR.size, NaN)
    
        aXR = XR[:, 0]
        aYR = YR[:, 0]     
    
        for i in range(0, Nalt * Nbeams):
            dd = min(np.sqrt( np.power(XR - mlonR[i], 2.0) / np.power(Dlon, 2.0) + np.power(YR - mlatR[i], 2.0) / np.power(Dlat, 2.0)))[0]
            tmin = np.sqrt( np.power(XR - mlonR[i], 2.0) / np.power(Dlon, 2.0) + np.power(YR - mlatR[i], 2.0) / np.power(Dlat, 2.0)).argmin()
        
            [amin, bmin] = ind2sub([x.size, y.size], tmin)
        
            fm0[i, 0] = Ae[i, 2]
            fm0[i, 1] = Ae[i, 0]
            fm0[i, 2 + amin] = Ae[i, 1]
      
            if ~np.isnan(mlonR[i] + mlatR[i]):
            
                ai = interpolate.RectBivariateSpline( (mlatR[i] + yy)[0:, 0],(mlonR[i] + xx)[0, 0:], zz, kx = 1, ky = 1)
                a = ai.ev( aYR, aXR )
                iia = np.nonzero(a)[0]
            
                ttx[i, iia] = a[iia] * Ae[i, 0]
                tty[i, iia] = a[iia] * Ae[i, 1]
         
        fm = np.zeros((Nalt * Nbeams, Nphi))
        fm = - (np.dot(ttx, GRx) + np.dot(tty, GRy))

    
        ### IMAGE MASK    
    
        ImCol = np.asmatrix(np.nanmax(np.sqrt(np.power(ttx, 2.0) + np.power(tty, 2.0)), axis = 0 ))

        Im = np.asarray(np.asmatrix(np.reshape(ImCol.H, (x.size, y.size), order = 'F')))    

        indices = np.transpose(np.where(Im > 0.15))
        indices = indices[np.argsort(indices[:,1])]
        

        k = spatial.ConvexHull(indices)
    
        ########if you would like to graph the hull, turn this on:
        #graph convex hull and points
        #plt.plot(indices[:,0], indices[:,1], 'o')
        #for simplex in k.simplices:
        #    plt.plot(indices[simplex,0], indices[simplex,1], 'k-')
        #plt.show()
 
        Imask = getMask(indices, k.simplices, x.size, y.size)
    
        #######if you would like to plot the mask, turn this on:
        #fig = plt.figure()
        #ax = fig.add_subplot(1,1,1)
        #plt.imshow(Imask, interpolation = 'nearest', cmap = plt.cm.ocean)
        #plt.show()
   
   
        #########################################################################################################
        ########################################## Compute G matrices ###########################################
        #########################################################################################################
   
        ### G (REGULARIZATION) MATRIX
   
        print '\nG1 matrix computation \n'
        (Guni, gamma0uni) = makeGuni(0, y.size, 2, 10, 2)
        #GuniInverse = scipy.linalg.inv(Guni)    
        GuniInverse = truncatedSVDinversion(Guni)[0]
        #########################################################################################################
    
    
        print 'G2 matrix computation'   
        tmask = np.reshape(Imask, (Nphi, 1), order = 'F')
        (G, Ginv) = Gmatrix3_mask(y.size, x.size, tmask, masksc, 1, derivWeights)
    
        #########################################################################################################
    
  
        #Resolve
        alphin = np.nan
        alphin1 = np.nan
        alphin2 = np.nan
    
    
        #########################################################################################################
        ############################################## Resolve E field ##########################################
        #########################################################################################################
    
    
        #choose good measurements
        tv = np.reshape(tv, tv.size, order = 'F')
        tdv = np.reshape(tdv, tdv.size, order = 'F')
        tsnr = np.reshape(tsnr, tsnr.size, order = 'F')
        ElR = np.reshape(ElR, ElR.size, order = 'F')
    

        AeCol1 = Ae[:, 0]
        AeCol2 = Ae[:, 1]
        AeCol3 = Ae[:, 2]
        
        print 'IokInput', IokInput

        ################### OLD
        #RHV added SNR
        # if IokInput == 0:
        #     with np.errstate(invalid='ignore'):       
        #         Iok = np.where(( (np.isfinite(AeCol1)) & (np.isfinite(AeCol2)) & (np.isfinite(AeCol3)) &       (np.isfinite(tv)) & (np.isfinite(tdv)) & (np.less(tdv, maxDv)) & (np.less(np.absolute(tv), maxV)) & (np.less(ElR, maxEl)) & (np.greater(tsnr,minSNR))))[0]
                
        # else:
        #     Iok = IokInput
        ################### END OLD

        if IokInput == 0:
            with np.errstate(invalid='ignore'):       
                Iok1 = np.where(( (np.isfinite(AeCol1)) & (np.isfinite(AeCol2)) & (np.isfinite(AeCol3)) & (np.isfinite(tv)) & (np.isfinite(tdv)) & (np.less(tdv, maxDv)) & (np.less(np.absolute(tv), maxV)) & (np.less(ElR, maxEl)) & (np.greater(tsnr,minSNR))))[0]
            
            #lonely points check
            Iok=[]
            for ii in Iok1:
                Iok2=Iok1[Iok1!=ii]
                sortdist=np.sort(np.sqrt((mlatR[ii]-mlatR[Iok2])**2.+(mlonR[ii]-mlonR[Iok2])**2.).ravel())
                if sortdist[1]<1.0: #second furthest point is less than 1 degree away
                    Iok.append(ii)
                else:
                    print 'rejecting lonely point, ',ii
                    print sortdist
        else:
            Iok = IokInput

        #gut check
        #print 'tdv check:'
        #for i in range(tdv.shape[0]):
        #    print i,tv[i],tdv[i],tsnr[i],(i in Iok)

    
        #RHV hack to skip empty entries
        if len(Iok)==0:
            continue
            
            
    
        tv = tv[Iok] * BabsR[Iok, 0]
        tdv = tdv[Iok] * BabsR[Iok, 0]
        tA = Ae[Iok, :]
        tfm = fm[Iok, :]
        tfm0 = fm0[Iok, :]
        c = np.diag(np.power(tdv, 2.0))
        tc = c[:]
        tfm1 = tfm[:]
    

        
        #########################################################################################################
        ### constant field solution
        print 'constant field solution'
        cinv  = truncatedSVDinversion(c, eps = 7.0e-7)[0]

    
        if dovpar: 
            E0 = np.zeros((tA.shape[1] , 1))
            E0 = np.dot(np.dot( np.dot(np.linalg.inv(np.dot(np.dot(tA.T, cinv), tA)) , tA.T), cinv), tv) #inv(tA'*cinv*tA)*tA'*cinv*tv
            
    
            covar0 = np.zeros((tA.shape[1], tA.shape[1]))
            covar0 = np.linalg.inv(np.dot(np.dot(tA.T, cinv), tA)) #inv(tA'*cinv*tA)
            
        
        else:
            ttA = np.zeros((tA.shape[0], 2))
            ttA = tA[:, 0:2]
        
            E0 = np.zeros((ttA.shape[1], 1))
            E0 = np.dot( np.dot( np.dot(np.linalg.inv(np.dot(np.dot(ttA.T, cinv), ttA)), ttA.T), cinv), tv) #inv(ttA'*cinv*ttA)*ttA'*cinv*tv
            E0 = np.hstack((E0, 0)) #set third entry in E0 to zero. need to do this after the inversion; otherwise singular. 
        
            covar0 = np.zeros((ttA.shape[1], ttA.shape[1]))
            covar0 = np.linalg.inv(np.dot(np.dot(ttA.T, cinv), ttA)) #inv(ttA'*cinv*ttA)
        
        dE0 = np.zeros(covar0.shape)
        Efor0 = np.zeros(np.dot(tA, E0).shape)
        
        
        
        #turn constant field solution on and off
        use0d = 1
        if use0d:
            print "E0 = " + np.array_str(E0)
    
            dE0 = np.sqrt(np.diag(covar0))
            print "dE0 = " + np.array_str(dE0) + "\n"
            
            Efor0 = np.dot(tA, E0)
        
        else:
            E0 = np.array([0, 0, 0])
            dE0 = np.array([0, 0, 0])
        Efor0 = np.reshape(Efor0, (Efor0.size, 1), order = 'F')
    
        tv = np.reshape(tv, (tv.size, 1), order = 'F')

        


 
        #########################################################################################################    
        #### 1D field solution
        print "1D field solution"
        tv1 = np.zeros(tv.shape)
        tv1 = tv - Efor0  
    

    
        condNumberInVec = np.empty((1, 1)) * alphin1    
        (E1, covarOut, alph01, alph1) = resolveEfield(GuniInverse, tfm0, tv1, c, condNumberInVec, use1d)

        dE1 = np.sqrt(np.diagonal(covarOut))
        tE1 = np.asarray(list(E1))
        tExConst = np.median(tE1[0:y.size])
        tE1[0:y.size] = tExConst
        Efor1 = np.dot(tfm0, tE1)
        tEfor1 = np.dot(tfm0, E1)

        Ex1 = E0[0] + E1[0: y.size]
        Ey1 = E0[1] + E1[y.size:]
        dEx1 = dE0[0] + dE1[0: y.size]
        dEy1 = dE0[1] + dE1[y.size: ]
    
        #########################################################################################################    
        ###If 1d solution converged, do chi square test to determine whether 2d solution is necessary. 
        if doChiSquare and not all(E1) == 0: 
            (chiSquare1D, p1D) = scipy.stats.chisquare(tv - Efor0, Efor1)
    
        
            ##if 1d + constant field solution is good enough, skip the 2d calculation to save time.    
            if p1D > 0.05 and not np.isnan(p1D):
                print "\n1D solution passes Chi-Square test. 2D solution will not be calculated. Plotting 1D + constant field results only."
                print "Chi Square = " + str(chiSquare1D) + ". P value = " + str(p1D) + ". \n"
                use2d = 0   
            
        #########################################################################################################
        ##2d calculation. Use2d is passed to resolveEfield and if it is zero, the function immediately returns zeros.     

        if use2d: print '\n2D field solution'
        tv1 = np.reshape(tv1, (tv1.size, 1), order = 'F')
        tv2 = tv1 - Efor1 * use1d
    
        condNumberInVec = np.empty((1, 1)) * alphin1

        # THIS RETURNS POTENTIAL AND COV OF POTENTIAL (see equ 17 and 19 from Nicolls 2014)
        (sol, covarOut, alph02, alph2) = resolveEfield(Ginv, tfm1, tv2, tc, condNumberInVec, use2d) 

        Efor2 = np.dot(tfm1, sol)
        sol =  np.reshape(sol, (sol.size, 1), order = 'F')
    
        Phi = np.reshape(sol, (x.size, y.size), order = 'F')
        dPhi = np.reshape(np.sqrt(np.diagonal(covarOut)), (x.size, y.size), order = 'F')

        #########################################################################################################
        ######################################### Reshape and plot results ######################################
        #########################################################################################################

        Etot = np.zeros((x.size, y.size, 2))
        dEtot = np.zeros((x.size, y.size, 2))

        #GRx = GRx.toarray()
        #GRy = GRy.toarray()
    
        Ey1 = np.reshape(Ey1, (Ey1.size, 1), order = 'F')
        dEy1 = np.reshape(dEy1, (dEy1.size, 1), order = 'F')
        # print((E0[1] * (1 - use1d)).shape)
        # print((np.matlib.repmat(Ey1, 1, y.size) * use1d).shape)
        # print((np.reshape( - np.dot(GRy, sol), (x.size, y.size), order = 'F') * use2d).shape)
        Etot[:, :, 0] =  E0[0] * (1 - use1d) + np.matlib.repmat(Ex1, 1, x.size).T * use1d + np.reshape( - np.dot(GRx, sol), (x.size, y.size), order = 'F') * use2d
    
        Etot[:, :, 1] = E0[1] * (1 - use1d) + np.matlib.repmat(Ey1, 1, x.size).T * use1d + np.reshape(- np.dot(GRy, sol), (x.size, y.size), order = 'F') * use2d
    
        sqrtDiagGyCGy =  np.reshape(np.sqrt(np.diagonal( np.dot(np.dot(GRy, covarOut), GRy.T))), (x.size * y.size, 1), order = 'F') * use2d
        
        dEtot[:, :, 0] = dE0[0] + use2d * np.reshape(np.sqrt(np.diagonal( np.dot(np.dot(GRx, covarOut), GRx.T) )), (x.size, y.size), order = 'F')
        dEtot[:, :, 1] = np.matlib.repmat(dEy1, 1, x.size).T + use2d * np.reshape(  sqrtDiagGyCGy  , (x.size, y.size), order = 'F')
    
    
        tEx = np.reshape(Etot[:, :, 0], (Nphi, 1), order = 'F')
        tEy = np.reshape(Etot[:, :, 1], (Nphi, 1), order = 'F')
    
        tdivE = np.dot(GRx, tEx) + np.dot(GRy, tEy)
        divE = np.reshape(tdivE, (x.size, y.size), order = 'F')
    
        divE[0:2, :] = NaN
        divE[x.size - 2: x.size, :] = NaN
        divE[:, 0:2] = NaN
        divE[:, y.size - 2: y.size] = NaN
    
        Etot[0:2, :, :] = NaN
        Etot[x.size - 2 : x.size, :, : ] = NaN
        Etot[:, 0:2, :] = NaN
        Etot[:, y.size - 2: y.size, : ] = NaN
    
        dEtot[0:2, :, :] = NaN
        dEtot[ x.size - 2 : x.size, :, :] = NaN
        dEtot[:, 0:2, :] = NaN
        dEtot[:, y.size - 2: y.size, : ] = NaN

        Phi[0:2, :] = NaN
        Phi[x.size - 2 : x.size, :] = NaN
        Phi[:, 0:2] = NaN
        Phi[:, y.size - 2: y.size] = NaN
    
        dPhi[0:2, :] = NaN
        dPhi[ x.size - 2 : x.size, :] = NaN
        dPhi[:, 0:2] = NaN
        dPhi[:, y.size - 2: y.size] = NaN  
        
        print "------------------------------------------------"
        print "Errors on E fit for time record " + str(timeSlice)
        print "\n"
        print "X: Max: " + str(np.nanmax(dEtot[:,:,0])) + ", Min: " + str(np.nanmin(dEtot[:,:,0])) + ", Mean: " + str(np.nanmean(dEtot[:,:,0]))
        print "Y: Max: " + str(np.nanmax(dEtot[:,:,1])) + ", Min: " + str(np.nanmin(dEtot[:,:,1])) + ", Mean: " + str(np.nanmean(dEtot[:,:,1]))
        print "------------------------------------------------"
        print "\n"
        
    
    
        if not all(Efor2) == 0:
            (chiSquare2D, p2D) = scipy.stats.chisquare(tv - Efor0 - Efor1, Efor2)
            print "Chi Square = " + str(chiSquare2D) + ". P value = " + str(p2D) + ". \n"
    
        results = {}
        results["x"] = x
        results["y"] = y
        results["X"] = X
        results["Y"] = Y
        results["mlonR"] = mlonR
        results["mlatR"] = mlatR
        results["simName"] = simName
        results["Efor0"] = Efor0
        results["Efor1"] = Efor1
        results["Efor2"] = Efor2
        results["use1d"] = use1d
        results["tv"] = tv
        results["tEfor1"] = tEfor1
        results["BabsFixed"] = BabsFixed
        results["Iok"] = Iok
        results["Ex1"] = Ex1
        results["Ey1"] = Ey1
        results["dEx1"] = dEx1
        results["dEy1"] = dEy1
        results["Imask"] = Imask
        results["use2d"] = use2d    
    
        results["simFlag"] = simFlag
        results["realFlag"] = realFlag
        results["masksc"] = masksc
        results["derivWeights"] = derivWeights
        results["gridSizeX"] = gridSizeX
        results["gridSizeY"] = gridSizeY
        results["hardCode"] = hardCode
        results["pathToData"] = pathToData
        results["label"] = label
        results["Perr"] = Perr

        results["Phi"] = Phi
        results["dPhi"] = dPhi
   
        results["Etot"] = 1.0e3 * Etot
        results["dEtot"] = 1.0e3 * dEtot
        results["Emag"] = 1.0e3 * np.sqrt(np.sum(np.power(Etot, 2.0), 2))
        results["Ex"] = 1.0e3 * Etot[:, :, 0]
        results["Ey"] =  1.0e3 * Etot[:, :, 1]
        results["dEx"] = 1.0e3 * dEtot[:, :, 0]
        results["dEy"] =  1.0e3 * dEtot[:, :, 1]
        results["DivE"] = divE


        # if simFlag:
        #     trueVlosR = np.reshape(trueVlosR, trueVlosR.size) ###SIMULATION ONLY#####
        #     trueVlos = np.reshape(trueVlosR[Iok] * BabsFixed, ((trueVlosR[Iok] * BabsFixed).size)) ###SIMULATION ONLY#####
        
        #     results["EtotTrue"] = EtotTrue
        #     results["trueVlos"] = trueVlos
        
        #     plotSim(x, y, X, Y, Etot, EtotTrue, mlonR, mlatR, simName, Efor0, Efor1, Efor2, use1d, tv, tEfor1, trueVlos, BabsFixed, Iok, divETrue, divE, Ex1, Ey1, Imask, use2d)
        
        # if realFlag:    
        #     results["timeSlice"] = timeSlice
            
        #     #RHV added simPath
        #     plotReal(x, y, X, Y, Etot, dEtot, tdv, mlonR, mlatR, simName, simPath, Efor0, Efor1, Efor2, use1d, tv, tEfor1, BabsFixed, Iok, divE, Ex1, Ey1, Imask, use2d)
        
        #RHV replace this savemat with an HDF write
        scipy.io.savemat(simPath + "/" + label, results)
        
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
        labeldict[clockTime] = label        

        
    return X, Y, Etotdict, mlonR, mlatR, Iokdict, divEdict, Imaskdict, labeldict

    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
