#lost in translation! 
#MATLAB functions that don't have exact analog in numpy or python

import numpy as np
from PIL import Image, ImageDraw
from math import sqrt
import matplotlib.pyplot as plt


def truncatedSVDinversion(matrixToInvert, truncate = 1, eps = 2.2204e-16):
    
    U, S, V = np.linalg.svd(matrixToInvert)
    condNum = S[0]/S[-1]
    
    if truncate == 0:
        pseudoInverse = np.dot(np.dot(V.T, np.diag(1.0/S)), U.T)
        return pseudoInverse, 0, condNum
        

    else: 
        numRows = matrixToInvert.shape[0]
        numCols = matrixToInvert.shape[1]
        sBiggest = S[0]
        #thresh = 0.5 * sqrt(numRows + numCols + 1.0) * eps * sBiggest
        thresh = np.max(matrixToInvert.shape) * sBiggest * eps
        
        Sinv = np.zeros(S.size)
        numTruncated = 0
    
        for index in range(0, S.size):
            if S[index] < thresh:
                Sinv[index] = 0.0
                numTruncated += 1
            else:
                Sinv[index] = 1.0/S[index]
            
        pseudoInverse = np.dot(np.dot(V.T, np.diag(Sinv)), U.T)
    
        return pseudoInverse, numTruncated, condNum
   
    

def getMask(indices, convexHullSimplices, maskWidth, maskHeight):
    
    #generating an image mask corresponding to the polygon inside the convex hull
    #get coordinates of convex hull     
    xCoords = np.reshape(indices[convexHullSimplices, 0], indices[convexHullSimplices, 0].size)
    yCoords = np.reshape(indices[convexHullSimplices, 1], indices[convexHullSimplices, 1].size)
              
    #since the polygon is always convex, averaging the points will always yield a point within the polygon
    originX = np.average(xCoords)
    originY = np.average(yCoords)
    
    #in order to draw the polygon to get the mask, we will need the coordinate pairs to be ordered either clockwise or counter-clockwise. 
    #this is easiest to do in polar coordinates by sorting by theta (JZW)!
    #convert to polar coordinates
    rCoords = np.zeros(xCoords.size)
    thetaCoords = np.zeros(xCoords.size)
    rCoords = np.sqrt(np.power((xCoords - originX), 2.0) + np.power((yCoords - originY), 2.0))
    thetaCoords = np.arctan2(yCoords - originY, xCoords - originX)
    #sort coordinates by increasing theta (going around the polygon)
    sortedIndices = np.argsort(thetaCoords)
    #re-order Cartesian coordinates using the order found for increasing theta
    xCoords = xCoords[sortedIndices].tolist()
    yCoords = yCoords[sortedIndices].tolist()
    
    #create an array of tuples of coordinate pairs ((x1, y1), (x2, y2), .... etc)
    Coords = zip(xCoords, yCoords)
    
    #create a matrix of zeros and fill in the polygon with 1s
    img = Image.new('L', (maskWidth, maskHeight), 0)
    ImageDraw.Draw(img).polygon(Coords, outline = 0, fill = 1)
    
    #the mask is the resulting matrix of 0s and 1s, with 1s corresponding to everything inside the convex hull!
    mask = np.array(img)  
    mask = np.transpose(np.asmatrix(mask))
    
    return mask

def MATLABdot(A, B):
    if A.shape == B.shape:
        
        numRow = A.shape[0]
        numCol = A.shape[1]
        
        C = np.zeros((1, numCol))
        
        for i in range(0, numCol):
            for j in range(0, numRow):
                C[0, i] += A[j, i] * B[j, i]
        
        return C
        
    else:
        print "Error. Matrix sizes do not match."
        return None 
        


#http://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python      
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    return ind
def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (cols, rows)      
        
def ind2sub2(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)         
        
        
#taken from http://stackoverflow.com/questions/4692196/discrete-laplacian-del2-equivalent-in-python user423805

def del2(M, stepX, stepY):
  dx = stepX
  dy = stepY
  rows, cols = M.shape
  dx = dx * np.ones ((1, cols - 1))
  dy = dy * np.ones ((rows - 1, 1))

  mr, mc = M.shape
  D = np.zeros ((mr, mc))

  if (mr >= 3):
      ## x direction
      ## left and right boundary
      D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:,0] * dx[:,1])
      D[:, mc-1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc-1]) \
          / (dx[:,mc - 3] * dx[:,mc - 2])
   
      ## interior points
      tmp1 = D[:, 1:mc - 1]
      tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
      tmp3 = np.kron (dx[:,0:mc -2] * dx[:,1:mc - 1], np.ones ((mr, 1)))
      D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3

  if (mr >= 3):
      ## y direction
      ## top and bottom boundary
      D[0, :] = D[0,:]  + \
          (M[0, :] - 2 * M[1, :] + M[2, :] ) / (dy[0,:] * dy[1,:])
   
      D[mr-1, :] = D[mr-1, :] \
          + (M[mr-3,:] - 2 * M[mr-2, :] + M[mr-1, :]) \
          / (dy[mr-3,:] * dx[:,mr-2])
   
      ## interior points
      tmp1 = D[1:mr-1, :]
      tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr-2, :])
      tmp3 = np.kron (dy[0:mr-2,:] * dy[1:mr-1,:], np.ones ((1, mc)))
      D[1:mr-1, :] = tmp1 + tmp2 / tmp3
   
  return D / 4