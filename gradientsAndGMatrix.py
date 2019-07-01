import sys
import numpy as np
import inspect
import scipy
from lostInTranslation import truncatedSVDinversion
from math import sqrt

def Gmatrix3_mask(M,N, mask, masksc, doInv, derivativeWeightDict):
    
    # Computes G = G0+G1+G2 where GN is a regularization matrix of electric
    # potential derivatives of order N. 
    #
    # M: 
    #   square-matrix dimension. G will be M^2xM^2 in size.
    # mask: 
    #   a column vector which should be M^2x1 in size containing 0's for cells
    #   that are masked and 1's for cells that are unmasked.
    # masksc: 
    #   a 3x2 array that sets the scaling factor for each G matrix for each
    #   order derivative (0-2, rows) where the mask is 0 (1st column) and where
    #   the mask is 1 (second column)
    # bndLevs:
    #   a 3x1 array that sets the boundary levels for each order derivative.  A
    #   low value will generally result in unconstrained boundaries, and large
    #   values will force the boundary potential to 0.  **bndLevs(1) (N=0) is
    #   not currently used.
    # doInv:
    #   whether to return the inverse of G; otherwise, Gi is a copy of G
    # derivativeWeightDict:
    #   a dictionary of weighting coefficients for each seperate term in the G matrix. The G matrix can be composed of up to 14 different terms:
        #shorthand notation: x denotes d/dx Phi and y denotes d/dy Phi (where Phi is the potential). xy denotes d/dx d/dy Phi. xyx denotes d^2/dxdy d/dx Phi. etc
        # 0-th order: Ex, Ey, or, in terms of the potential, d/dx Phi and d/dy Phi. in notation: x and y.
        # 1-st order: d/dx Ex, d/dx Ey, d/dy Ex, d/dy Ey, or in shorthand notation: xx, xy, yx, yy.
        # 2-nd order: xxx, xxy, xyx, yxx, xyy, yxy, yyx, yyy
        #The shorthand labels (x, xy, xyx, etc) are the keys for the weight dictionaries, and the values are the weights for the corresponding terms. This enables the user to weight derivatives of different components of the potential in different directions differently. To revert to the old masking style where the weights only applied to 0th, 1st, and 2nd order magnitudes and did not distinguish between components and directions, all keys with equal number of characters (i.e. x and y for 0th order, xx, xy, yx, yy for 1st order, etc) must be set to the same value. 

    #M-column M-row matrix is mapped into a vector by sampling columnwise
    #(going down a column, then moving to the next column).
    #gamma applied to this vector gives the laplacian where going along a row
    #is advancing in x with step dx, and going down a column is advancing in y
    #with step dy
    
    
    #unpack derivative weights dictionary
    x = derivativeWeightDict['x']
    y = derivativeWeightDict['y']
    xy = derivativeWeightDict['xy']
    yx = derivativeWeightDict['yx']
    xx = derivativeWeightDict['xx']
    yy = derivativeWeightDict['yy']
    xxx = derivativeWeightDict['xxx']
    xxy = derivativeWeightDict['xxy']
    yyx = derivativeWeightDict['yyx']
    yyy = derivativeWeightDict['yyy']
    
    
    
    if M < 3:
        G = 'error'
        Ginv = 'error'
        print 'Error: M is too small.'
        sys.exit() 
    
    #MM = pow(M, 2)
    MM = M*N
    
    #A transforms the vector into what you would have got if
    #the sampling of the matrix had been rowwise. Multiplying by A again transforms back.
    A = np.zeros((MM, MM))
    for nn in range(0, MM):
        A[nn, :] = getRow_nn_transposeOper(nn, M, N)
    
    
    #Gamm_b1 and Phi_b take the first derivative in y for one column, except at end points.
    #Gamm_b2 takes the second derivative in y for one column, except at end points.
    Gamma_b2 = np.zeros((M, M))
    Phi_b = np.zeros((M, M))
    Gamma_b1 = np.zeros((M, M))
    
    for it in range(0, M - 1):
        Gamma_b1[it, it : it + 2] = [2.0, -2.0]
    
    for it in range(1, M - 1):
        Gamma_b2[it, it - 1 : it + 2] = [1.0, -2.0, 1.0]
        Phi_b[it, it - 1: it + 2] = [1.0, 0.0, -1.0]
        
    #Gamma_b1[:] = Phi_b
    
    Phi_b[0, 0 : 2] = [2.0, -2.0]    
    Phi_b[M - 1, (M - 3) : M] = [0.0, 2.0, -2.0]
    
    Gamma0 = np.identity(MM)
    Gamma1 = np.zeros((MM, MM))
    Gamma2 = np.zeros((MM, MM))
    Phi1 = np.zeros((MM, MM))
    
    for it in range(1, M + 1):
        Gamma1[ (it - 1) * M  : (it-1) * M + M, (it - 1) * M : (it - 1) * M + M] = Gamma_b1
        Gamma2[ (it - 1) * M  : (it-1) * M + M, (it - 1) * M : (it - 1) * M + M] = Gamma_b2
        Phi1[   (it - 1) * M  : (it-1) * M + M, (it - 1) * M : (it - 1) * M + M] = Phi_b
        
    G = np.zeros((MM, MM))
    
    
    #magnitude of electric field
    imask = 0    
    
    tmask = np.asarray(mask * (masksc[imask , 1] - masksc[imask, 0]) + masksc[imask, 0], dtype = np.float64)
    Gamma_mm = np.dot(Gamma0, Phi1) * tmask
    Gamma_mn = np.dot(Gamma0, np.dot(A, np.dot(Phi1, A))) * tmask
    Gamma_nm = np.dot(np.dot(Gamma0, A), Phi1) * tmask
    Gamma_nn = np.dot(Gamma0, np.dot(Phi1, A)) * tmask
    
    Gamma_mm_t = np.transpose(Gamma_mm)
    Gamma_mn_t = np.transpose(Gamma_mn)
    Gamma_nm_t = np.transpose(Gamma_nm)
    Gamma_nn_t = np.transpose(Gamma_nn)
    
    G += x * np.dot(Gamma_mm_t, Gamma_mm) + y * np.dot(Gamma_mn_t, Gamma_mn) + x * np.dot(Gamma_nm_t, Gamma_nm) + y * np.dot(Gamma_nn_t, Gamma_nn)
    
    
    
    #1st derivative of electric field
    imask = 1
    tmask = np.asarray(mask * (masksc[imask, 1] - masksc[imask, 0]) + masksc[imask, 0], dtype = np.float64)
    
    Gamma_mm = np.dot(Gamma1, Phi1) * tmask
    Gamma_mn = np.dot(Gamma1, np.dot(A, np.dot(Phi1, A))) * tmask
    Gamma_nm = np.dot(np.dot(Gamma1, A), Phi1) * tmask
    Gamma_nn = np.dot(Gamma1, np.dot(Phi1, A)) * tmask
    
    Gamma_mm_t = np.transpose(Gamma_mm)
    Gamma_mn_t = np.transpose(Gamma_mn)
    Gamma_nm_t = np.transpose(Gamma_nm)
    Gamma_nn_t = np.transpose(Gamma_nn)
    
    G += xx * np.dot(Gamma_mm_t, Gamma_mm) + xy * np.dot(Gamma_mn_t, Gamma_mn) + yx * np.dot(Gamma_nm_t, Gamma_nm) + yy * np.dot(Gamma_nn_t, Gamma_nn)
    
    


    #2nd derivative of electric field
    imask = 2
    tmask = np.asarray(mask * (masksc[imask, 1] - masksc[imask, 0]) + masksc[imask, 0], dtype = np.float64)
    
    Gamma_mm = np.dot(Gamma2, Phi1) * tmask
    Gamma_mn = np.dot(Gamma2, np.dot(A, np.dot(Phi1, A))) * tmask
    Gamma_nm = np.dot(np.dot(Gamma2, A), Phi1) * tmask
    Gamma_nn = np.dot(Gamma2, np.dot(Phi1, A)) * tmask
    
    Gamma_mm_t = np.transpose(Gamma_mm)
    Gamma_mn_t = np.transpose(Gamma_mn)
    Gamma_nm_t = np.transpose(Gamma_nm)
    Gamma_nn_t = np.transpose(Gamma_nn)
    
    G += xxx * np.dot(Gamma_mm_t, Gamma_mm) + xxy * np.dot(Gamma_mn_t, Gamma_mn) + yyx * np.dot(Gamma_nm_t, Gamma_nm) + yyy * np.dot(Gamma_nn_t, Gamma_nn)

    



    if doInv:
        
        #U, s, V = np.linalg.svd(G)
        #print "Regular SVD inversion"
        #GiSVD = np.dot(np.dot(V.T, np.diag(1.0/s)),U.T)
        #print 'G reciprocal condition number = ' + np.array_str(s[s.size - 1]/s[0]) + '\n'
        
        print "Truncated SVD inversion"
        GiSVD, numTrunc, condNum = truncatedSVDinversion(G)
        print "Number of singular values truncated = " + str(numTrunc)
        print 'G reciprocal condition number = ' + str(condNum) + '\n'
        
        

    else:
        Gi = G
        
    return G, GiSVD

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def makeGuni(Nzeros, M, Mrep, bVal, order):
    
    if order == 1: 
        inmat = [-0.5, 0.0, 0.5]
        dm = 1
    elif order == 2:
        inmat = [1.0, -2.0, 1.0]
        dm = 1
    elif order == 22:
        inmat = [-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0]
        dm = 2
        
    MM = M * Mrep + Nzeros
    
    G = np.zeros((MM, MM))
    gamma0uni = np.zeros((MM, MM))
    
    for it0 in range( dm , M * Mrep - dm ):
        it = it0 + Nzeros
        gamma0uni[it, it - dm : it + dm + 1] = inmat
    
    for io in range(0, Nzeros):
        gamma0uni[io, io] = 1/M
        
    for ii in range(1, Mrep + 1):
        for io in range(1, dm + 1):                        
            gamma0uni[(M * (ii - 1) + dm + Nzeros) - 1, (M * (ii - 1) + dm + Nzeros) - 1] = bVal
            gamma0uni[(M * (ii) - dm + 1 + Nzeros) - 1, (M * (ii) - dm + 1 + Nzeros) - 1] = bVal
            
        ##################################################################################################################
        #gamma0uni[(M * (ii - 1) + dm + Nzeros) - 1, (M * (ii - 1) + dm + Nzeros) - 1] = bVal
        #gamma0uni[(M * (ii) - dm + 1 + Nzeros) - 1, (M * (ii) - dm + 1 + Nzeros) - 1] = bVal
        ##################################################################################################################
    
    gamma0transpose = np.asarray(np.asmatrix(gamma0uni).H)
    G = np.dot(gamma0transpose, gamma0uni)
      
    return G, gamma0uni
    
    
    
    
    
def getGradientOperators(N, M):

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
    
    