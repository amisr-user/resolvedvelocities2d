# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:00:02 2017

@author: rvarney
"""

import numpy
import matplotlib.pyplot as plt
import os
import datetime

import io_utils

runnames=['20170118.011_lp_30sec-fitcal_2dVEF_001001',
          '20170118.013_lp_30sec-fitcal_2dVEF_001001',
          '20170118.015_lp_30sec-fitcal_2dVEF_001001',
          '20170118.017_lp_30sec-fitcal_2dVEF_001001',
          '20170118.019_lp_30sec-fitcal_2dVEF_001001',
          '20170118.021_lp_30sec-fitcal_2dVEF_001001',
          '20170118.011_lp_1min-fitcal_2dVEF_001001',
          '20170118.013_lp_1min-fitcal_2dVEF_001001',
          '20170118.015_lp_1min-fitcal_2dVEF_001001',
          '20170118.017_lp_1min-fitcal_2dVEF_001001',
          '20170118.019_lp_1min-fitcal_2dVEF_001001',
          '20170118.021_lp_1min-fitcal_2dVEF_001001',
          '20170118.011_lp_2min-fitcal_2dVEF_001001',
          '20170118.013_lp_2min-fitcal_2dVEF_001001',
          '20170118.015_lp_2min-fitcal_2dVEF_001001',
          '20170118.017_lp_2min-fitcal_2dVEF_001001',
          '20170118.019_lp_2min-fitcal_2dVEF_001001',
          '20170118.021_lp_2min-fitcal_2dVEF_001001']
          
for runname in runnames:
    print runname
    d=io_utils.read_whole_h5file(runname+'.h5')
    
    plt.rcParams['figure.figsize']=10,10
    plt.rcParams['font.size']=16
    
    for r in range(d['/Fit2D']['DivE'].shape[0]):
        fig=plt.figure()
        div_clrs=plt.pcolormesh(d['/Grid']['x'][0,:],d['/Grid']['y'][0,:],d['/Fit2D']['DivE'][r,:,:],vmin=-1e-5,vmax=1e-5,cmap='RdBu_r')
        qv=plt.quiver(d['/Grid']['x'][0,:],d['/Grid']['y'][0,:],d['/Fit2D']['Ex'][r,:,:],d['/Fit2D']['Ey'][r,:,:],d['/Fit2D']['Emag'][r,:,:],scale=2500,cmap='bone_r')
        qk = plt.quiverkey(qv, 0.5, 0.875, 100, '100 mV/m',
                            labelpos='E',
                            coordinates='figure',
                            fontproperties={'weight': 'bold'}) 
        plt.title(datetime.datetime(1970,1,1)+datetime.timedelta(seconds=int(d['/Time']['UnixTime'][r,0])))
        plt.xlabel('Mag Longitude')
        plt.ylabel('Mag Latitude')
        
        hc=plt.colorbar(div_clrs)
        hc.set_label('DivE')
        
        plt.savefig('./figtmp/fig_'+runname+'_'+str(r)+'.png',format='png',dpi=100)
        plt.close(fig)
        
    os.system('ffmpeg -r 4 -i figtmp/fig_'+runname+'_%d.png -c:v libx264 -pix_fmt yuv420p ' + runname + '_divE.mp4')