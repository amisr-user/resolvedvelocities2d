import numpy as np
from pylab import *
from matplotlib.colors import LogNorm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import scipy.io
import os

def plotTotRadAng(Etot, Eradial, Eangular, mlonR, mlatR, X, Y, Iok, timeStamp):
    
    figure(figsize = (24, 7))    
    with np.errstate(invalid='ignore'):    

    ######1
        ax1 = subplot(1, 3, 1)
        Q1 = quiver( X, Y, Etot[:, :, 0], Etot[:, :, 1], color = 'r', scale = 0.5) #width = 0.003,
        plt.scatter(mlonR[Iok], mlatR[Iok], s = 2)
        
        ax2 = subplot(1, 3, 2)
        Q2 = quiver( X, Y, Eradial[:, :, 0], Eradial[:, :, 1], color = 'r', scale = 0.5) #width = 0.003,
        plt.scatter(mlonR[Iok], mlatR[Iok], s = 2)
        
        ax3 = subplot(1, 3, 3)
        Q3 = quiver( X, Y, Eangular[:, :, 0], Eangular[:, :, 1], color = 'r', scale = 0.5) #width = 0.003,
        plt.scatter(mlonR[Iok], mlatR[Iok], s = 2)

    figName = "/run results/EtotRadAng_%s.jpg"  %(timeStamp)
    
    
    savefig(figName)
        
def plotReal(x, y, X, Y, Etot, dEtot, tdv, mlonR, mlatR, simName, out_dir, Efor0, Efor1, Efor2, use1d, tv, tEfor1, BabsFixed, Iok, divE, Ex1, Ey1, Imask, use2d = 1):    
    #RHV added out_dir
    folderName = simName.split('_')[0] + '_' + simName.split('_')[1]

    #sloppy making new folders...
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    #RHV re-direct the output appropriately
    fig1name = out_dir+"/%s-fig1.eps" %(simName)
    fig2name = out_dir+"/%s-fig2.eps" %(simName)
    fig3name = out_dir+"/%s-fig3.eps" %(simName)
    fig4name = out_dir+"/%s-fig4.eps" %(simName)

    '''
    fig1name = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/%s/%s/%s-fig1.eps" %(folderName, simName, simName)
    fig2name = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/%s/%s/%s-fig2.eps" %(folderName, simName, simName)
    fig3name = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/%s/%s/%s-fig3.eps" %(folderName, simName, simName)
    fig4name = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/%s/%s/%s-fig4.eps" %(folderName, simName, simName)
    '''
    with np.errstate(invalid='ignore'):    

    ######1
        figure(figsize = (12, 12))
        Q1 = quiver( X, Y, Etot[:, :, 0], Etot[:, :, 1], color = 'r', width = 0.003)
        plt.scatter(mlonR[Iok], mlatR[Iok], s = 2)
        savefig(fig1name)



    ######2
        figure(figsize = (20, 13))
    
        if use2d:
            tfor = Efor0 + Efor1 * use1d + Efor2 

        ###SUBPLOT 1
        ax1 = subplot(4,1,1)    

        #graph
        plot(tv, color = 'b', label = 'Meas.')
        if use2d:
            plot(tfor, color = 'r', label = '2d')
        plot(Efor0, color = 'g', label = '0d')
        plot(Efor0 + tEfor1, color = 'y', label = '1d')
        plt.xlim(0, Efor0.size)


        #plt.xlim(np.min(trueVlos[0]), np.max(trueVlos[0]))
        #plt.ylim(np.min(Y), np.max(Y))

        #legend
        box1 = ax1.get_position()
        ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
        plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), prop = {'size':12})

        #ticks
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks)
        ax1.yaxis.set_major_locator(yloc)

        ax1.set_ylabel('LOS E')


        ###SUBPLOT2
        ax2 = subplot(4,1,2)

        #graph
        plot(tv - Efor0, color = 'b', label = 'Meas. -0d')
        plot(Efor1, color = 'r', label = '1dm')
        plot(tEfor1, color = 'g', label = '1d')
        plt.xlim(0, Efor1.size)


        #legend
        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), prop = {'size':12})

        #ticks
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks)
        ax2.yaxis.set_major_locator(yloc)



        ###SUBPLOT 3
        ax3 = subplot(4,1,3)

        #graph
        plot(tv - Efor0 - Efor1 * use1d, color = 'b', label = 'Meas. -1d')
        if use2d:
            plot(Efor2, 'r', label = '2d')
            plt.xlim(0, Efor2.size)
        else:
            plt.xlim(0, Efor1.size)

        #legend
        box3 = ax3.get_position()
        ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), prop = {'size':12})

        #ticks
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks)
        ax3.yaxis.set_major_locator(yloc)

        #ax3.set_xlabel('Measurement')
        
        ###SUBPLOT 4
        ax4 = subplot(4,1,4)

        #graph
        tdv = np.reshape(tdv, (tdv.size, 1))
    
        plot((np.power((tv - tfor), 2.0) / np.power(tdv, 2.0)), color = 'b', label = 'Chi Square')
        plt.xlim(0, tv.size)

        #legend
        box4 = ax4.get_position()
        ax4.set_position([box4.x0, box4.y0, box4.width * 0.8, box4.height])
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), prop = {'size':12})

        #ticks
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks)
        ax4.yaxis.set_major_locator(yloc)

        ax4.set_xlabel('Measurement')
        
        savefig(fig2name)

    #######2.5
        figure(figsize = (10, 10))
        #chiSquare = np.zeros()
        #print Iok
        #figure(figsize = (12, 12))
        #Q1 = quiver( X, Y, Etot[:, :, 0], Etot[:, :, 1], color = 'r', width = 0.003)
        plt.scatter(mlonR[Iok], mlatR[Iok], s = 2)

    #######3

        #figure(figsize = (20 + 1, 2 * 13/3 + 1))
        figure(figsize = (20 + 1, 13/3 + 1))
        
        Emag = 1.0e3 * np.sqrt(np.sum(np.power(Etot, 2.0), 2))
        Ex = 1.0e3 * Etot[:, :, 0]
        Ey = 1.0e3 * Etot[:, :, 1]
        
        EmagErr = 1.0e3 * np.sqrt(np.sum(np.power(dEtot, 2.0), 2))
        ExErr = 1.0e3 * dEtot[:, :, 0]
        EyErr = 1.0e3 * dEtot[:, :, 1]
        
        EmagChiSq = np.power(Emag - EmagErr, 2.0) / np.power(EmagErr, 2.0)
        ExChiSq = np.power(Ex - ExErr, 2.0) / np.power(ExErr, 2.0)
        EyChiSq = np.power(Ey - EyErr, 2.0) / np.power(EyErr, 2.0)
        
        
        Emag[Emag < 0] = 0.0
        Emag[np.iscomplex(Emag)] = 0.0

        
        vmin1 = np.nanmin(Emag)
        vmax1 = np.nanmax(Emag)
        vmin2 = np.nanmin(Ex) 
        vmax2 = np.nanmax(Ex)
        vmin3 = np.nanmin(Ey)
        vmax3 = np.nanmax(Ey)
        vmin4 = np.nanmin(divE)
        vmax4 = np.nanmax(divE)
        
        vmin5 = np.nanmin(EmagChiSq)
        vmax5 = np.nanmax(EmagChiSq)
        vmin6 = np.nanmin(ExChiSq)
        vmax6 = np.nanmax(ExChiSq)
        vmin7 = np.nanmin(EyChiSq)
        vmax7 = np.nanmax(EyChiSq)
        
        

        #SUBPLOT 1
       # ax1 = subplot(2, 4, 1)
        ax1 = subplot(1, 4, 1)
        orig_cmap = matplotlib.cm.hot
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 0.05, name = 'shifted')
        plt.pcolor(X, Y, Emag, cmap = orig_cmap, vmin = vmin1 - (vmax1 - vmin1) * 0.05, vmax = vmax1)
        cb1 = plt.colorbar()
        
        orig_cmap = matplotlib.cm.seismic
        
    
    
        plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        plt.title('Magnitude E')
        ax1.set_ylabel('Mag. Lat.')
        ax1.set_xlabel('Mag. Lon.')
        #plt.grid()
        
    

        #SUBPLOT 2
        #ax2 = subplot(2, 4, 2)
        ax2 = subplot(1, 4, 2)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax2)/(np.absolute(vmax2) + np.absolute(vmin2))), name = 'shifted')
        plt.pcolor(X, Y, Ex, cmap = shifted_cmap, vmin = vmin2, vmax = vmax2)
        cb2 = plt.colorbar()
        plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        plt.title('E_x')
        ax2.set_xlabel('Mag. Lon.')
        


        #SUBPLOT 3
        #ax3 = subplot(2, 4, 3)
        ax3 = subplot(1, 4, 3)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax3)/(np.absolute(vmax3) + np.absolute(vmin3))), name = 'shifted')
        plt.pcolor(X, Y, Ey, cmap = shifted_cmap, vmin = vmin3, vmax = vmax3)
        cb3 = plt.colorbar()
        plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        plt.title('E_y')
        ax3.set_xlabel('Mag. Lon.')
        



        #SUBPLOT 4
        #ax4 = subplot(2, 4, 4)
        ax4 = subplot(1, 4, 4)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax4)/(np.absolute(vmax4) + np.absolute(vmin4))), name = 'shifted')
        plt.pcolor(X, Y, divE, cmap = shifted_cmap, vmin = vmin4, vmax = vmax4)
        cb4 = plt.colorbar() #format='%.0e'
        plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        plt.title('Div E')
        
        #savefig(fig3name)
        
       
        ##SUBPLOT 5
        #ax5 = subplot(2, 4, 5)
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax5)/(np.absolute(vmax5) + np.absolute(vmin5))), name = 'shifted')
        #plt.pcolor(X, Y, EmagChiSq, cmap = shifted_cmap, vmin = vmin5, vmax = vmax5)
        #cb5 = plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        #plt.xlim(np.min(X), np.max(X))
        #plt.ylim(np.min(Y), np.max(Y))
        #plt.title('E Magnitude Chi Squared')
        #ax5.set_xlabel('Mag. Lon.')
        
        #SUBPLOT 6
        #ax6 = subplot(2, 4, 6)
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax6)/(np.absolute(vmax6) + np.absolute(vmin6))), name = 'shifted')
        #plt.pcolor(X, Y, ExChiSq, cmap = shifted_cmap, vmin = vmin6, vmax = vmax6)
        #cb6 = plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        #plt.xlim(np.min(X), np.max(X))
        #plt.ylim(np.min(Y), np.max(Y))
        #plt.title('E x Chi Squared')
        #ax6.set_xlabel('Mag. Lon.')
        
        #SUBPLOT 7
        #ax7 = subplot(2, 4, 7)
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax7)/(np.absolute(vmax7) + np.absolute(vmin7))), name = 'shifted')
        #plt.pcolor(X, Y, EyChiSq, cmap = shifted_cmap, vmin = vmin7, vmax = vmax7)
        #cb7 = plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        ##plt.xlim(np.min(X), np.max(X))
        #plt.ylim(np.min(Y), np.max(Y))
        #plt.title('E y Chi Squared')
        #ax7.set_xlabel('Mag. Lon.')


        #plt.tight_layout()


   
        
    #######3.5 (temperature)

        #figure(figsize = (6, 6))

        #SUBPLOT 1
        #ax1 = subplot(1, 1, 1)
        #plt.pcolor(X, Y, ionTemp, cmap = 'seismic', vmin = np.nanmin(ionTempColVec), vmax = np.nanmax(ionTempColVec))
        #cb1 = plt.colorbar()
        #clim1 = (cb1.vmin, cb1.vmax)
    
    
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        #plt.xlim(np.min(X), np.max(X))
        #plt.ylim(np.min(Y), np.max(Y))
        #plt.title('Ion Temperature')
        #ax1.set_ylabel('Mag. Lat.')
        #ax1.set_xlabel('Mag. Lon.')
        
        
        #fig35name = "%s/%s-fig35_ionTemperature.eps" %(simName, simName)
        #savefig(fig35name)
        

    #######4
        figure(figsize = (6, 6))

        #SUBPLOT 1
        ax1 = subplot(1, 2, 1)
            

        t = Etot[:, :, 0]
        t[Imask == 0] = NaN


        for i in range(0, t.shape[1]):
            if use2d:
                plot(1.0e3 * t[:, i], y.T, color = 'r')
        plot(1.0e3 * Ex1, y.T, color = 'k', linewidth = 2.0)

        ax1.set_xlabel('Ex')
        ax1.set_ylabel('Mag.Lat')

        #SUBPLOT 2
        ax2 = subplot(1, 2, 2)

        t = Etot[:, :, 1]
        t[Imask == 0] = NaN
      

        for i in range(0, t.shape[1] - 1):
            plot(1.0e3 * t[:, i], y.T, color = 'r')
        if use2d:
            plot(1.0e3 * t[:, t.shape[1] - 1], y.T, color = 'r', label = '2D')
        plot(1.0e3 * Ey1, y.T, color = 'k', linewidth = 2.0, label = '1D')

        ax2.set_xlabel('Ey')

        ax2.legend(bbox_to_anchor = (-0.12, 1.12), loc = 'upper center', ncol = 3)

        savefig(fig4name)




        #show()





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


def plotSim(x, y, X, Y, Etot, EtotTrue, mlonR, mlatR, simName, Efor0, Efor1, Efor2, use1d, tv, tEfor1, trueVlos, BabsFixed, Iok, divETrue, divE, Ex1, Ey1, Imask, masksc, use2d = 1):
    
    folderName = simName.split('_')[0] + '_' + simName.split('_')[1]


    fig1name = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/%s/%s/%s-fig1.png" %(folderName, simName, simName)
    fig2name = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/%s/%s/%s-fig2.png" %(folderName, simName, simName)
    fig3name = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/%s/%s/%s-fig3.png" %(folderName, simName, simName)
    fig4name = "/home/rvarney/AMISR/VEF/NAM_final/FINAL_VERSION/run results/%s/%s/%s-fig4.png" %(folderName, simName, simName)
    
    
    with np.errstate(invalid='ignore'):    
    
    ######1
        figure(figsize = (12, 12))
        Q1 = quiver( X, Y, Etot[:, :, 0], Etot[:, :, 1], color = 'r', width = 0.003)
        Q2 = quiver( X, Y, EtotTrue[:, :, 0], EtotTrue[:, :, 1], color = 'lime', width = 0.002)
        #plot(mlonR[Iok], mlatR[Iok], 'k.')
        plt.scatter(mlonR[Iok], mlatR[Iok], s = 2)
        savefig(fig1name)
    

    
    ######2
        figure(figsize = (20, 13))
        
        if use2d:
            tfor = Efor0 + Efor1 * use1d + Efor2 
    
        ###SUBPLOT 1
        ax1 = subplot(3,1,1)    
    
        #graph
        plot(tv, color = 'b', label = 'Meas.')
        if use2d:
            plot(tfor, color = 'r', label = '2d')
        plot(Efor0, color = 'g', label = '0d')
        plot(Efor0 + tEfor1, color = 'y', label = '1d')
        plot(trueVlos , color = 'k', label = 'True')
        plt.xlim(0, Efor0.size)
    
    
        #plt.xlim(np.min(trueVlos[0]), np.max(trueVlos[0]))
        #plt.ylim(np.min(Y), np.max(Y))
    
        #legend
        box1 = ax1.get_position()
        ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
        plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), prop = {'size':12})
    
        #ticks
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks)
        ax1.yaxis.set_major_locator(yloc)

        ax1.set_ylabel('LOS E')

    
        ###SUBPLOT2
        ax2 = subplot(3,1,2)
    
        #graph
        plot(tv - Efor0, color = 'b', label = 'Meas. -0d')
        plot(Efor1, color = 'r', label = '1dm')
        plot(tEfor1, color = 'g', label = '1d')
        plt.xlim(0, Efor1.size)
    
    
        #legend
        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), prop = {'size':12})
    
        #ticks
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks)
        ax2.yaxis.set_major_locator(yloc)
    
    
    
        ###SUBPLOT 3
        ax3 = subplot(3,1,3)
    
        #graph
        plot(tv - Efor0 - Efor1 * use1d, color = 'b', label = 'Meas. -1d')
        if use2d:
            plot(Efor2, 'r', label = '2d')
            plt.xlim(0, Efor2.size)
        else:
            plt.xlim(0, Efor1.size)
    
        #legend
        box3 = ax3.get_position()
        ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), prop = {'size':12})
    
        #ticks
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks)
        ax3.yaxis.set_major_locator(yloc)
    
        ax3.set_xlabel('Measurement')
    
    
        savefig(fig2name)
    


    #######3
    
    #######3

        #figure(figsize = (20 + 1, 2 * 13/3 + 1))
        figure(figsize = (20 + 1, 13/3 + 1))
        
        Emag = 1.0e3 * np.sqrt(np.sum(np.power(Etot, 2.0), 2))
        Ex = 1.0e3 * Etot[:, :, 0]
        Ey = 1.0e3 * Etot[:, :, 1]
        
        EmagTrue = 1.0e3 * np.sqrt(np.sum(np.power(EtotTrue, 2.0), 2))
        ExTrue = 1.0e3 * EtotTrue[:, :, 0]
        EyTrue = 1.0e3 * EtotTrue[:, :, 1]
        
       
        
        Emag[Emag < 0] = 0.0
        Emag[np.iscomplex(Emag)] = 0.0

        
        vmin1 = np.nanmin(EmagTrue)
        vmax1 = np.nanmax(EmagTrue)
        vmin2 = np.nanmin(ExTrue) 
        vmax2 = np.nanmax(ExTrue)
        vmin3 = np.nanmin(EyTrue)
        vmax3 = np.nanmax(EyTrue)
        vmin4 = np.nanmin(divETrue)
        vmax4 = np.nanmax(divETrue)
        
        vmin5 = np.nanmin(Emag)
        vmax5 = np.nanmax(Emag)
        vmin6 = np.nanmin(Ex) 
        vmax6 = np.nanmax(Ex)
        vmin7 = np.nanmin(Ey)
        vmax7 = np.nanmax(Ey)
        vmin8 = np.nanmin(divE)
        vmax8 = np.nanmax(divE) 
     
        orig_cmap = matplotlib.cm.RdBu_r
        


        figure(figsize = (20, 13))
    
        #SUBPLOT 1
        ax1 = subplot(3, 4, 1)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 0.0, name = 'shifted')
        plt.pcolor(X, Y, EmagTrue, cmap = 'Reds', vmin = vmin1, vmax = vmax1)
        cb1 = plt.colorbar()
        clim1 = (cb1.vmin, cb1.vmax)
        
        
        plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('|E| Simulated')
        ax1.set_ylabel('Magnetic latitude')
        #plt.grid()
    
        #SUBPLOT 2
        ax2 = subplot(3, 4, 2)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax2)/(np.absolute(vmax2) + np.absolute(vmin2))), name = 'shifted')
        plt.pcolor(X, Y, ExTrue, cmap = shifted_cmap, vmin = vmin2, vmax = vmax2)
        cb2 = plt.colorbar()
        clim2 = (cb2.vmin, cb2.vmax)
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('E_x (True)')
        
 
    
    
        #SUBPLOT 3
        ax3 = subplot(3, 4, 3)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1.0, name = 'shifted')
        plt.pcolor(X, Y, EyTrue, cmap = 'Blues_r', vmin = vmin3, vmax = vmax3)
        cb3 = plt.colorbar()
        clim3 = (cb3.vmin, cb3.vmax)
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('E_y (True)')
        

    
    
    
        #SUBPLOT 4
        ax4 = subplot(3, 4, 4)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax4)/(np.absolute(vmax4) + np.absolute(vmin4))), name = 'shifted')
        plt.pcolor(X, Y, divETrue, cmap = shifted_cmap, vmin = vmin4, vmax = vmax4)
        cb4 = plt.colorbar(format='%.0e')
        clim4 = (cb4.vmin, cb4.vmax)
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('Div E (True)')
    
        
    
    
    
        #SUBPLOT 5
        ax5 = subplot(3, 4, 5)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 0.0, name = 'shifted')
        plt.pcolor(X, Y, Emag, cmap = 'Reds', vmin = vmin5, vmax = vmax5)
        plt.clim(clim1)
        plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('Magnitude E (Pred)')
        ax5.set_ylabel('Magnetic latitude')
    
    
        #SUBPLOT 6
        ax6 = subplot(3, 4, 6)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax2)/(np.absolute(vmax2) + np.absolute(vmin2))), name = 'shifted')
        plt.pcolor(X, Y, Ex, cmap = shifted_cmap, vmin = vmin2, vmax = vmax2)  
        #plt.clim(clim2)
        plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('E_x (Pred)')
    

        #SUBPLOT 7
        ax7 = subplot(3, 4, 7)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1.0, name = 'shifted')
        plt.pcolor(X, Y, Ey, cmap = 'Blues_r', vmin = vmin3, vmax = vmax3)
        #plt.clim(clim3)
        plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
       # plt.title('E_y (Pred)')
    
    
    
        #SUBPLOT 8
        ax8 = subplot(3, 4, 8)
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax4)/(np.absolute(vmax4) + np.absolute(vmin4))), name = 'shifted')
        plt.pcolor(X, Y, divE, cmap = shifted_cmap, vmin = vmin4, vmax = vmax4)
        #plt.clim(clim4)
        plt.colorbar(format='%.0e')
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('Div E (Pred)')
        
    
    
    
    
        diffMag = 1.0e3 * np.sqrt(np.sum(np.power(EtotTrue, 2.0), 2)) - 1.0e3 * np.sqrt(np.sum(np.power(Etot, 2.0), 2))
        diffX = 1.0e3 * EtotTrue[:, :, 0] - 1.0e3 * Etot[:, :, 0]
        diffY = 1.0e3 * EtotTrue[:, :, 1] - 1.0e3 * Etot[:, :, 1]
        diffDIV = divETrue - divE
        
        diffMagMasked = np.zeros(diffMag.shape)
        diffXMasked = np.zeros(diffX.shape)
        diffYMasked = np.zeros(diffY.shape)
        diffDIVMasked = np.zeros(diffDIV.shape)
        
        for i in range(0,diffMag.shape[0]):
            for j in range(0,diffMag.shape[1]):
                diffMagMasked[i,j] = Imask[i,j] * diffMag[i,j]
                diffXMasked[i,j] = Imask[i,j] * diffX[i,j]
                diffYMasked[i,j] = Imask[i,j] * diffY[i,j]
                diffDIVMasked[i,j] = Imask[i,j] * diffDIV[i,j]
                
        diffMagMasked[diffMagMasked == 0] = np.nan
        diffXMasked[diffXMasked == 0] = np.nan
        diffYMasked[diffYMasked == 0] = np.nan
        diffDIVMasked[diffDIVMasked == 0] = np.nan
        
        MagMax = np.round(np.nanmax(np.abs(diffMag)), decimals = 1)
        MagMin = np.round(np.nanmin(np.abs(diffMag)), decimals = 1)
        MagMean = np.round(np.nanmean(np.abs(diffMag)), decimals = 1)
        MagMedian = np.round(np.nanmedian(np.abs(diffMag)), decimals = 1)
        MagMaskMax = np.round(np.nanmax(np.abs(diffMagMasked)), decimals = 1)
        MagMaskMin = np.round(np.nanmin(np.abs(diffMagMasked)), decimals = 1)
        MagMaskMean = np.round(np.nanmean(np.abs(diffMagMasked)), decimals = 1)
        MagMaskMedian = np.round(np.nanmedian(np.abs(diffMagMasked)), decimals = 1)
        
        XMax = np.round(np.nanmax(np.abs(diffX)), decimals = 1)
        XMin = np.round(np.nanmin(np.abs(diffX)), decimals = 1)
        XMean = np.round(np.nanmean(np.abs(diffX)), decimals = 1)
        XMedian = np.round(np.nanmedian(np.abs(diffX)), decimals = 1)
        XMaskMax = np.round(np.nanmax(np.abs(diffXMasked)), decimals = 1)
        XMaskMin = np.round(np.nanmin(np.abs(diffXMasked)), decimals = 1)
        XMaskMean = np.round(np.nanmean(np.abs(diffXMasked)), decimals = 1)
        XMaskMedian = np.round(np.nanmedian(np.abs(diffXMasked)), decimals = 1)
        
        YMax = np.round(np.nanmax(np.abs(diffY)), decimals = 1)
        YMin = np.round(np.nanmin(np.abs(diffY)), decimals = 1)
        YMean = np.round(np.nanmean(np.abs(diffY)), decimals = 1)
        YMedian = np.round(np.nanmedian(np.abs(diffY)), decimals = 1)
        YMaskMax = np.round(np.nanmax(np.abs(diffYMasked)), decimals = 1)
        YMaskMin = np.round(np.nanmin(np.abs(diffYMasked)), decimals = 1)
        YMaskMean = np.round(np.nanmean(np.abs(diffYMasked)), decimals = 1)
        YMaskMedian = np.round(np.nanmedian(np.abs(diffYMasked)), decimals = 1)
        
        DIVMax = (np.nanmax(np.abs(diffDIV)))
        DIVMin = (np.nanmin(np.abs(diffDIV)))
        DIVMean = (np.nanmean(np.abs(diffDIV)))
        DIVMedian = (np.nanmedian(np.abs(diffDIV)))
        DIVMaskMax = (np.nanmax(np.abs(diffDIVMasked)))
        DIVMaskMin = (np.nanmin(np.abs(diffDIVMasked)))
        DIVMaskMean = (np.nanmean(np.abs(diffDIVMasked)))
        DIVMaskMedian = (np.nanmedian(np.abs(diffDIVMasked)))
        
        print str(masksc) + "    |    " + "mag  " + "    |    " + "  Ex" + "    |    " + "    Ey" + "    |    " + "     divergence"
        print "unmasked:     " + str(MagMin) + "-" + str(MagMax) + "   |   " + str(XMin) + "-" + str(XMax) + "   |   " + str(YMin) + "-" + str(YMax) + "   |   " + str(DIVMin) + "-" + str(DIVMax) 
        print "masked:     " + str(MagMaskMin) + "-" + str(MagMaskMax) + "   |   " + str(XMaskMin) + "-" + str(XMaskMax) + "   |   " + str(YMaskMin) + "-" + str(YMaskMax) + "   |   " + str(DIVMaskMin) + "-" + str(DIVMaskMax) 
        print "unmasked mean: " + str(MagMean) + "   |   " + str(XMean) + "   |   " + str(YMean) + "   |   " + str(DIVMean) 
        print "masked mean: " + str(MagMaskMean) + "   |   " + str(XMaskMean) + "   |   " + str(YMaskMean)  + "   |   " + str(DIVMaskMean)
        print "unmasked median: " + str(MagMedian) + "   |   " + str(XMedian) + "   |   " + str(YMedian) + "   |   " + str(DIVMedian) 
        print "masked median: " + str(MagMaskMedian) + "   |   " + str(XMaskMedian) + "   |   " + str(YMaskMedian)  + "   |   " + str(DIVMaskMedian) 
        
        
        
        vmin9 = np.nanmin(EmagTrue - Emag)
        vmax9 = np.nanmax(EmagTrue - Emag)
        vmin10 = np.nanmin(ExTrue - Ex)
        vmax10 = np.nanmax(ExTrue - Ex)
        vmin11 = np.nanmin(EyTrue - Ey)
        vmax11 = np.nanmax(EyTrue - Ey)
        vmin12 = np.nanmin(divETrue - divE)
        vmax12 = np.nanmax(divETrue - divE)
        
  
    
        #SUBPLOT 9
        ax9 = subplot(3, 4, 9)
        diffMag = EmagTrue - Emag
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 0.5, name = 'shifted')
        plt.pcolor(X, Y, diffMag, cmap = shifted_cmap, vmin = -30, vmax = 30)
        #plt.clim((np.nanmin(diffMag), np.nanmax(diffMag)))
        plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('Magnitude E (Diff.)')
        ax9.set_xlabel('Magnetic longitude')
        ax9.set_ylabel('Magnetic latitude')
    
    

    
    
        #SUBPLOT 10
        ax10 = subplot(3, 4, 10)
        diffX = ExTrue - Ex
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 0.5, name = 'shifted')
        plt.pcolor(X, Y, diffX, cmap = shifted_cmap, vmin = -22, vmax = 22)
        #plt.clim((np.nanmin(diffX), np.nanmax(diffX)))
        plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('E_x (Diff.)')
        ax10.set_xlabel('Magnetic longitude')
    
    
    
    

    
        #SUBPLOT 11
        ax11 = subplot(3, 4, 11)
        diffY = EyTrue - Ey
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 0.5, name = 'shifted')
        plt.pcolor(X, Y, diffY, cmap = shifted_cmap, vmin = -22, vmax = 22)
        #plt.clim((np.nanmin(diffY), np.nanmax(diffY)))
        plt.colorbar()
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('E_y (Diff.)')
        ax11.set_xlabel('Magnetic longitude')
    
       
    
        #SUBPLOT 12
        ax12 = subplot(3, 4, 12)
        diffDIV = divETrue - divE
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(vmax4)/(np.absolute(vmax4) + np.absolute(vmin4))), name = 'shifted')
        plt.pcolor(X, Y, diffDIV, cmap = shifted_cmap, vmin = vmin4, vmax = vmax4)
        #plt.clim((np.nanmin(diffDIV), np.nanmax(diffDIV)))
        plt.colorbar(format='%.0e')
        #plt.scatter(mlonR, mlatR, color = 'k', s = 2)
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        #plt.title('Div E (Diff.)')
        ax12.set_xlabel('Magnetic longitude')
        
    
        #plt.tight_layout()

    
        savefig(fig3name)
        #savefig("fig3.png")
    
    
    #######4
        figure(figsize = (6, 6))
    
        #SUBPLOT 1
        ax1 = subplot(1, 2, 1)
    
        t = Etot[:, :, 0]
        t[Imask == 0] = NaN
        t1 = EtotTrue[:, :, 0]
        t1[Imask == 0] = NaN
    
        for i in range(0, t1.shape[1]):
            plot(1.0e3 * t1[:, i], y.T, color = 'gray')
            if use2d:
                plot(1.0e3 * t[:, i], y.T, color = 'r')
        plot(1.0e3 * Ex1, y.T, color = 'k', linewidth = 2.0)
    
        ax1.set_xlabel('Ex')
        ax1.set_ylabel('Mag.Lat')
    
        #SUBPLOT 2
        ax2 = subplot(1, 2, 2)
    
        t = Etot[:, :, 1]
        t[Imask == 0] = NaN
        t1 = EtotTrue[:, :, 1]
        t1[Imask == 0] = NaN
    
        for i in range(0, t1.shape[1] - 1):
            plot(1.0e3 * t1[:, i], y.T, color = 'gray')
            plot(1.0e3 * t[:, i], y.T, color = 'r')
        plot(1.0e3 * t1[:, t1.shape[1] - 1], y.T, color = 'gray', label = 'True')
        if use2d:
            plot(1.0e3 * t[:, t1.shape[1] - 1], y.T, color = 'r', label = '2D')
        plot(1.0e3 * Ey1, y.T, color = 'k', linewidth = 2.0, label = '1D')

        ax2.set_xlabel('Ey')

        ax2.legend(bbox_to_anchor = (-0.12, 1.12), loc = 'upper center', ncol = 3)
    
        savefig(fig4name)
    
    

    
        #show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
