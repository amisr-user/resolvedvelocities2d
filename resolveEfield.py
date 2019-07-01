from math import *
#import numpy as np
import numpy as np
#import bottleneck as bn
import scipy
from lostInTranslation import truncatedSVDinversion
from decimal import *

FLOATING_POINT_ACCURACY =  2.2204e-16

# Uhhhh, this doesn't actually resolve the electric field, it returns the potential:
# Equations, 17 and 19 from Nicolls 2014
def resolveEfield(Ginv, M, V, C, condNumberInVec, useResults = 1):
    
    # Inputs:
    # Ginv = inverse regularization matrix
    # M = measurement matrix
    # V = vector of measurements
    # C = covariance matrix of measurements
    # condNumberInVec = vector of past condition numbers for
    
    u_svd, s_svd, v_svd = np.linalg.svd(C)
    sBiggest = s_svd[0]
    thresh = np.max(C.shape) * sBiggest * FLOATING_POINT_ACCURACY
    
    if useResults == 0:
        sol = np.zeros((Ginv.shape[0], 1))
        covarOut= np.zeros((Ginv.shape))
        alph0 = 0 
        alph = 0 
        
        return sol, covarOut, alph0, alph
        
        
    else: 
    
        (alph0, Nit, condNumberOut, failcode) = getRegularizationParameter(Ginv, M, V, C, thresh)

    
        condNumberInVec = np.asarray([0.0]) #disables condition number smoothing
    

        if (np.max(condNumberInVec.shape) > 4):
            if not np.isnan(np.sum( condNumberInVec[-5: condNumberInVec.size, 0])):
                condNumberToUse =  np.nanmedian(np.vstack((condNumberInVec, condNumberOut)))
            
                (alph, Nit, condNumberToUse, failcode2) = getRegularizationParameter(Ginv, M, V, C, condNumberToUse)
        
            else:
                condNumberToUse = condNumberOut
                alph = alph0
        else: 
            condNumberToUse = condNumberOut
            alph = alph0
    
        if failcode > 0:
            sol = np.zeros((Ginv.shape[0], 1))
            covarOut = np.zeros((Ginv.shape))
        
        else:
            print "Nit: " + str(Nit) + "; alph0: " + str(alph0) + "; alphToUse: " + str(alph) + "; condiNumber0: " + str(condNumberOut) + "; condiNumberToUse: " + str(condNumberToUse)
            sol = getSolution(alph, Ginv, M, V, C)
            #posterior covariance matrix
            sE = Ginv / alph
            covarOut = sE - np.dot(np.dot(sE, M.T) , np.linalg.solve( np.dot(np.dot(M, sE), M.T) + C , np.dot(M, sE)) )
        
        return sol, covarOut, alph0, alph




def getSolution(alph, Ginv, M, V, C):
    
    Nmeas = V.size
    bet = 2.0 * sqrt(Nmeas)/alph
    tmat = np.dot(np.dot(M, Ginv), M.T) / 2.0
    lda = np.linalg.lstsq((sqrt(Nmeas)/bet * C + tmat) , V)[0]
   
    sol = np.dot(Ginv, np.dot(M.T, lda)) / 2.0    
    return sol
    

    
    
def getRegularizationParameter(Ginv, M, V, C, thresh, conditionNumberGoal = float('nan')):
    
    #Solves for alpha, the regularization parameter, using Newton's method.

    # Inputs:
    # Ginv = inverse regularization matrix
    # M = measurement matrix
    # V = measurements
    # C = covariance matrix of meeasurements
    # conditionNumberGoal = If this is entered the regularization parameter will be set to achieve the desired condition number, instead of to the curvature minimization criteria.



    # constants
    tol = 1.0e-8 #tolerance for convergence 
    maxIterations = 1000 #maximum number of iterations; will return NaN if solution does not converge
    Nmeas = V.size
    
    #initial guess
    tmat = np.dot(np.dot(M, Ginv), M.T) / 2.0
    
    pinvTMAT = truncatedSVDinversion(tmat)[0]    #0.5 * C * 0.0 + tmat originally
    lda = np.dot(pinvTMAT, V)
    bet = np.sqrt(np.dot(np.dot(lda.T, C), lda))

    testValue = np.dot(np.dot(V.T, scipy.linalg.inv(C)), V) / Nmeas

    cC = np.linalg.cond(C)
    ctmat = np.linalg.cond(tmat) 
    
    
    if isnan(conditionNumberGoal):
        conditionNumberOn = 0
        conditionNumberGoal = 1.0e15 #In this case this is the max condition number that will be accepted.
    else:
        conditionNumberOn = 1  
        
        
        

    Nit = 0
    failcode = 0
    DONE = 0
    
    while not DONE: 
        Nit += 1

        if Nit > maxIterations:
            print 'alph failed to converge; exceeded maximum number of iterations: ' + str(fabs(lastPositive - lastNegative) / bet)
            bet = float('nan')
            alph = float('nan')
            condNumberOut = float('nan')
            failcode = 2
            break
            
        elif (testValue - 1) < 0:
            print 'alph failed to converge because measurement error is too large'
            bet = float('nan')
            alph = float('nan')
            condNumberOut = float('nan')
            failcode = 1
            break
            
        ### check to see whether sqrt(Nmeas)/bet * C (\alpha \times C) divided by tmat (M G^{-1} M^{T}) is smaller than the threshold value. If it is, the errors are so small that C is going to zero and \alpha is going to infinity, and the numerical solution will not work. 
        
        if np.median(abs(((sqrt(Nmeas)/bet * C) / (tmat)))[np.nonzero(abs(((sqrt(Nmeas)/bet * C) / (tmat))))]) < FLOATING_POINT_ACCURACY:
            print "alpha failed to converge because alpha * C negligible compared to MG^{-1}M^T."
            bet = float('nan')
            alph = float('nan')
            condNumberOut = float('nan')
            failcode = 3
            break
        
        H = sqrt(Nmeas)/bet * C + tmat
        
        
        if not conditionNumberOn:
            lda = np.linalg.solve(H, V)
            f1 = pow(bet, 2.0)
            f2 = np.dot(np.dot(lda.T, C), lda)
        
        else: 
            if ctmat > cC:
                f1 = np.asscalar(np.linalg.cond(H))
                f2 = conditionNumberGoal
            
            else: 
                f2 = np.asscalar(np.linalg.cond(H))
                f1 = conditionNumberGoal
                
        
        if ('lastPositive' not in locals()) or ('lastNegative' not in locals()):
            if f1 > f2:
                lastPositive = bet
                bet = pow(10.0, log10(bet) - 1.0)
                
            else:
                lastNegative = bet
                bet = pow(10.0, log10(bet) + 1.0)
          
            if ('lastPositive' in locals()) and ('lastNegative' in locals()):
                bet = pow(10.0, (log10(lastPositive) + log10(lastNegative))  / 2.0)
                
             
        else:
            
            if f1 > f2:
                lastPositive = bet
                
            else:
                lastNegative = bet
            
            bet = sqrt( (pow(lastPositive, 2.0) + pow(lastNegative, 2.0)) / 2.0)
            
            
            if (fabs(lastPositive - lastNegative) / bet) < tol:
                condNumberOut = np.linalg.cond(H)
                
                
                if (condNumberOut <= conditionNumberGoal) or (conditionNumberOn):
                    DONE = 1
                    
                else: 
                    print 'alph converged but need to reduce the condition number for H, since it is ' + str(condNumberOut) 
                    conditionNumber = 1
                    bet = float('nan')
                    alph = float('nan')
                    condNumberOut = float('nan')
                    failcode = 1
                    break
                    
                    if Nit + 1 <= maxIterations:
                        del lastPositive
                        del lastNegative
                
        
    alph = 2.0 * sqrt(float(Nmeas)) / bet
        
            
    return alph, Nit, condNumberOut, failcode
    