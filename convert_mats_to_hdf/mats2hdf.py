# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:33:00 2017

@author: rvarney
"""

import numpy
import scipy.io
import os
import tables

import io_utils

datadir='/home/asreimer/projects/AMISR/git/resolvedvelocities2d/run_results/'
runname='20170302.002_lp_1min-cal_2dVEF_001001'

datapath=os.path.join(datadir,runname)
recStart=0

rec=recStart
matlist=[]
while os.path.exists(os.path.join(datapath,runname+'_%d'%rec)):
    try:
        matlist.append(scipy.io.loadmat(os.path.join(os.path.join(datapath,runname+'_%d'%rec),runname+'.mat'),struct_as_record=False))
        print 'loaded ',os.path.join(os.path.join(datapath,runname+'_%d'%rec),runname+'.mat')
    except IOError:
        #Fitter failed, use empty dict as place holder for 
        matlist.append({})
    rec+=1
    
nrecs=rec-recStart

#find the first non-empty mat to determine the sizes of all arrays
rec=0
while len(matlist[rec])==0:
    rec+=1
matf=matlist[rec]

#parse the mat in a structured way
grid={}
for key in ['x','y','X','Y','gridSizeX','gridSizeY']:
    grid[key]=matf[key]
    
options={}
for key in ['BabsFixed','hardCode','masksc','Perr','realFlag','simFlag','use1d','use2d']:
    options[key]=matf[key]
    
derivWeights={}
dW=matf['derivWeights'][0,0]
for key in dir(matf['derivWeights'][0,0]):
    if key[0]!='_':
        derivWeights[key]=eval('dW.'+key)
    
beamGeom={}
for key in ['mlatR','mlonR']:
    beamGeom[key]=matf[key]

losfit={}
for key in ['Efor0','Efor1','Efor2','tEfor1','tv']:    
    losfit[key]=numpy.nan*numpy.zeros(tuple([nrecs]+list(matf['mlatR'].shape)))
    
fit2d={}
for key in ['DivE','Emag','Etot','dEtot','Ex','Ey','Phi','dPhi']:
    fit2d[key]=numpy.nan*numpy.zeros(tuple([nrecs]+list(matf[key].shape)))
    
fit1d={}
for key in ['Ex1','Ey1']:
    fit1d[key]=numpy.nan*numpy.zeros(tuple([nrecs]+list(matf[key].shape)))
    
#grab time out of the original file
time=io_utils.read_partial_h5file(matf['pathToData'][0],['/Time'])['/Time']

for key in time.keys():
    time[key]=time[key][recStart:,:]

#parse all the other records
for rec in range(nrecs):
    if len(matlist[rec])>0:
        for key in fit2d.keys():
            fit2d[key][rec,:]=matlist[rec][key]
        for key in fit1d.keys():
            fit1d[key][rec,:]=matlist[rec][key]
        for key in losfit.keys():
            losfit[key][rec,matlist[rec]['Iok'][:,0]]=matlist[rec][key][0][0]

#write to hdf5 file
#outfile=tables.open_file(runname+'.h5', "w", driver="H5FD_CORE")
outfile=tables.open_file(runname+'.h5', "w", driver="H5FD_CORE")
io_utils.write_outputfile(outfile,grid,keys2do=grid.keys(),groupname='Grid')
io_utils.write_outputfile(outfile,options,keys2do=options.keys(),groupname='Options')
io_utils.write_outputfile(outfile,derivWeights,keys2do=derivWeights.keys(),groupname='DerivWeights')
io_utils.write_outputfile(outfile,beamGeom,keys2do=beamGeom.keys(),groupname='BeamGeom')
io_utils.write_outputfile(outfile,losfit,keys2do=losfit.keys(),groupname='LOSfit')
io_utils.write_outputfile(outfile,fit2d,keys2do=fit2d.keys(),groupname='Fit2D')
io_utils.write_outputfile(outfile,fit1d,keys2do=fit1d.keys(),groupname='Fit1D')
io_utils.write_outputfile(outfile,time,keys2do=time.keys(),groupname='Time')

outfile.close()
