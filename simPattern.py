from constants import *
import numpy as np
from lostInTranslation import del2
from pylab import *

def simPattern(x, y, X, Y, dx, dy, Dlon, Dlat): 
        
    #create a blob with [amplitude, width x, width y, location x, location y]
    blob = np.array([1.0e3, 25.0e3, 50.0e3, np.mean(x) + 1.0, np.mean(y) + 1.0]) 
    blob1 = np.array([-1.0e3, 50.0e3, 25.0e3, np.mean(x) - 1.0, np.mean(y) - 1.25])
    backPot = np.array([0.0e3, 25.0e3, 0.0e3, 0.0e3])
    #backPot = np.array([1.0e3, 25.0e3, 1.0e3, 1.0e3])
    
    #blob = np.array([2.0e3, 1750.0e3, 30.0e3, np.mean(x) + 2.0, np.mean(y) + 0.75]) 
    
    #backPot = np.array([0.1e3, 25.8e3, 0.1e3, 0.0e3])
        
    phiTrue = np.zeros((x.size, y.size))
    
    phiTrue = backPot[0] * (X - x[0, 0]) / (x[0, x.size - 1] - x[0, 0]) + backPot[1] * (Y - y[0, 0]) / (y[0, y.size - 1] - y[0, 0])
    
    #this line with this backPot does not do anything, but it could be placeholder in case of different backPot
    phiTrue = phiTrue + backPot[2] * np.power((X - x[0, 0]) / (x[0, x.size - 1] - x[0, 0]), 2.0) + backPot[3] * np.power((Y - y[0, 0]) / (y[0, y.size - 1] - y[0, 0]), 2.0)
    
    
    phiTrue = phiTrue + blob[0] * np.exp( -( np.power(X - blob[3] , 2.0) / 2.0 / np.power(blob[1]/ Dlon , 2.0)  + np.power(Y - blob[4], 2.0) / 2.0 / np.power(blob[2] / Dlat, 2.0)))
    
    phiTrue = phiTrue + blob1[0] * np.exp( -( np.power(X - blob1[3], 2.0) / 2.0 / np.power(blob1[1] / Dlon, 2.0) + np.power(Y - blob1[4], 2.0) / 2.0 / np.power(blob1[2] / Dlat, 2.0)))
    
    
    divETrue = - del2(phiTrue, dx, dy) * 4.0 

    Fy,Fx =  np.gradient(phiTrue, dy, dx)
    
    EtotTrue = np.zeros((x.size, y.size, 2))
    EtotTrue[:,:,0] = -Fx
    EtotTrue[:,:,1] = -Fy

    vVecTrue = np.zeros((x.size, y.size, 2))
    vVecTrue[:,:,0] = -EtotTrue[:,:,1]/BabsFixed;
    vVecTrue[:,:,1] = EtotTrue[:,:,0]/BabsFixed;
    
    #figure(figsize = (6, 13))
    #ax1 = subplot(2, 1, 1)
    #plt.pcolor(X, Y, 1.0e3 * np.sqrt(np.sum(np.power(EtotTrue, 2.0), 2)), cmap = 'ocean')
    
    #ax2 = subplot(2, 1, 2)
    #plt.pcolor(X, Y, 1.0e3 * phiTrue, cmap = 'ocean')
    #show()
    
    return divETrue, EtotTrue, vVecTrue