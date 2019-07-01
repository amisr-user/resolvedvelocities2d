import matplotlib 
matplotlib.use('Agg')
import pylab as plt
import pyfits # allows me to import in the pyfits packages
import numpy as np
from scipy.ndimage.interpolation import rotate
import os
import sys
import datetime
import scipy.io 
from plotProcess import shiftedColorMap
from getProjections import *
from plotProcess import plotTotRadAng

# original test files that this was developed under are located (as of 21 july 2014) at:




def read_fits(file):
    # read in the file
    f_red = pyfits.open(file)

    # import the data and the header
    head_red = f_red[0].header
    data_red = f_red[0].data

    # just to close this right away
    f_red.close()
    return head_red,data_red

def find_beam_pixels(data_az,data_el,az_in,el_in,dang):
    #dang = 1.0
    # test for the field aligned beam
    # az_fa = 360.-154.3
    # el_fa = 77.5
    # 
    # # I am going to use the where command to try to find the pixel within a degree 
    # # of field aligned az and field aligned elevation
    # # I need to plot this with respect to pixel location, NOT the actual lat and lon
    # 
    # q = np.where( (data_az >= (az_fa-1.0)) & (data_az <= (az_fa+1.0)) & 
    #           (data_el >= (el_fa-1.0)) & (data_el >= (el_fa+1.0)) 
    #           )
    # # q returns the pixels of interest
    # print q

    az = az_in
    el = el_in
    print az,el
    # if(az < 0):
#       az = 360.+az
    # I am going to use the where command to try to find the pixel within a degree 
    # of field aligned az and field aligned elevation
    # I need to plot this with respect to pixel location, NOT the actual lat and lon
    if(az >= 0):
        q = np.where( (data_az >= (az-dang)) & (data_az <= (az+dang)) & 
            (data_el >= (el-dang)) & (data_el >= (el+dang)))
    if(az < 0):
        print az-dang,az+dang,el-dang,el+dang
        q = np.where( (data_az > (az-dang)) & (data_az <= (az+dang)) & 
            (data_el >= (el-dang)) & (data_el >= (el+dang)))
    # q returns the pixels of interest
    print q
    az_out = q[0]
    el_out = q[1]
    return az_out,el_out

def find_FA_pixels(data_az,data_el,az_in,el_in):
    
    az = az_in
    el = el_in
    print az,el
    # if(az < 0):
#       az = 360.+az
    # I am going to use the where command to try to find the pixel within a degree 
    # of field aligned az and field aligned elevation
    # I need to plot this with respect to pixel location, NOT the actual lat and lon
    q = np.where((data_el >= (el-1.0)) & (data_el <= (el+1.0)) &  (data_az <= (1.0)))
    # q returns the pixels of interest
    print q
    az_out = q[0]
    el_out = q[1]
    return az_out,el_out


def azel_geo2mag(data_az_in,data_el_in,dec_in,incl_in):
    data_az = np.array(data_az_in)
    data_el = np.array(data_el_in)

    dec = np.deg2rad(dec_in)
    incl = np.deg2rad(incl_in)

    # eq 3 from Heinselman and Nicolls 2008
    E_geo = np.cos(np.deg2rad(data_el))*np.sin(np.deg2rad(data_az))
    N_geo = np.cos(np.deg2rad(data_el))*np.cos(np.deg2rad(data_az))
    U_geo = np.sin(np.deg2rad(data_el))

    # now apply the rotation matrix
    # eq 5 from Heinselman and Nicolls 2008
    E_mag = (E_geo*np.cos(dec))-(N_geo*np.sin(dec))
    A = (N_geo*np.cos(dec))+(E_geo*np.sin(dec))
    N_mag = (U_geo*np.cos(incl))+(np.sin(incl)*A)
    U_mag = (U_geo*np.sin(incl))-(np.cos(incl)*A)

    # now back out the angles
    az_mag = np.rad2deg(np.arctan2(E_mag,N_mag))
    R_mag = np.sqrt((E_mag*E_mag)+(N_mag*N_mag))
    el_mag = np.rad2deg(np.arctan2(U_mag,R_mag))
    return az_mag,el_mag




# plot all sky field aligned only
def plot_allsky_FA(file_in, dir_out, az_cal, el_cal, X, Y, mlonR, mlatR, Etot, Iok, divE, Imask, divEGlobalMax, divEGlobalMin, timeStamp):     
    
    
    
    
    if timeStamp == '853':
        Jdict = {}
        scipy.io.loadmat('/Users/nmaksimova/Desktop/NAM/VEF Reconstruction/FINAL VERSION/Jpar_withterms1204.mat', Jdict)
        Jparallel = Jdict["J"] * 1000000.0
        term1 = Jdict["sigP_term1"] * 1000000.0
        term2 = Jdict["gradSigP_term2"] * 1000000.0
        term3 = Jdict["minus_gradSigH_term3"] * 1000000.0
        mlon = Jdict["mlon"]
        mlat = Jdict["mlat"]
            
    
    mlonCoords = np.zeros(Iok.size)
    mlatCoords = np.zeros(Iok.size)
    
    
    for index in range(0, Iok.size):
        mlonCoords[index] = mlonR[Iok][0,index,0]
        mlatCoords[index] = mlatR[Iok][0,index,0]
        #mlonCoords[index] = mlonR[Iok][index,0]
        #mlatCoords[index] = mlatR[Iok][index,0]
        
    minIndex = np.argmin(mlatCoords)
    
    upBlat = mlatCoords[minIndex]
    upBlon = mlonCoords[minIndex]
    
    Eradial = np.zeros(Etot.shape)
    Eangular = np.zeros(Etot.shape)
    
    for i in range(0, Etot.shape[0]):
        for j in range(0, Etot.shape[1]):
            (Eradial[i,j,:], Eangular[i,j,:]) = getProjections((upBlon, upBlat), (X[i,j], Y[i,j]), (Etot[i,j,0], Etot[i,j,1]))
                       
    #plotTotRadAng(Etot, Eradial, Eangular, mlonR, mlatR, X, Y, Iok, str(timeStamp))
    
    #return
    
    
    
    # Purpose
    # This is the key routine
    # The function of this routine is basically to align the all sky image with the 
    # the field aligned beam from PFISR
    #The images are in terms of pixels, so I use the where command to get the index
    # that the beam corresponds to.  
    #Input
    #file_in = including directory string
    # dir_out = out directory location.  I name the file below
    #az_cal = azimuth calibration fits file.  This is an all sky grid corresponding 
    #to the azimuth angles for a given pixel
    #el_cal= Same as azimuth cal. A map of pixel locations with values of elevation
    #optional: VEF_fit_dict: include VEF reconstruction algorithm result in plot for this time record.
    # Important Note: The az and el files are in geographic coordinates, but the 
    #all sky images are in geomagnetic coordinates.  This was confirmed by Don. 
    # Important Note #2: I tried to do this using the lon/lat files on the website
    # when I plotted the beam lon/lat, it looked like nonsense so I abandoned it for the
    # time being
    # read in data
    head_az, data_az = read_fits(az_cal) #('PKR_DASC_20110112_AZ_10deg.FITS')
    head_el, data_el = read_fits(el_cal)#('PKR_DASC_20110112_EL_10deg.FITS')
    head_green, data_green = read_fits(file_in)

    # select only 5577 for the time being
    # the data being read in should be transposed because of the IDL/python problem of
    # rows being columns in python vs idl.
    # the 180 degrees is to put it into the same frame
    # This nominally aligns to the all sky movies found online.  It may not be perfect
    # but it is good enough.
    #data =rotate(np.transpose(data_green), angle=180)
    data = np.transpose(data_green)
    #convert the geographic az/el grid into geomagnetic coordinates
    # local field line at az = +22 and El = 77.5
    mag_az,mag_el =azel_geo2mag(data_az,data_el,22.,77.5) 
    
    #   rotate to put north at bottom of image and east to the right
    #   again, to be consistent with videos on website.
    mag_az = rotate(mag_az, angle=180)
    mag_el = rotate(mag_el,angle=180)
    
    fig = plt.figure(figsize = (18, 9))
    azEl_to_geoMag = scipy.io.loadmat(os.path.abspath('azEl_to_geoMag.mat'))
    data_az = azEl_to_geoMag['data_az']
    data_el = azEl_to_geoMag['data_el']
    mlatArr = azEl_to_geoMag['mlat']
    mlonArr = azEl_to_geoMag['mlon']
    
    data[np.where(data <= 0)] = 0.0
    
    plt.pcolormesh(mlonArr, mlatArr, data.T, vmin = 400, vmax = 700, cmap = 'ocean_r')
    #plt.colorbar(cmap = 'ocean_r')

    
    with np.errstate(invalid='ignore'):    

    ######1
        MagnitudeForColor = 1000.0 * np.sqrt(Etot[:, :, 0]*Etot[:, :, 0] + Etot[:, :, 1]* Etot[:, :, 1])
    
        print "X: Max: " + str(np.nanmax(Etot[:,:,0])) + ", Min: " + str(np.nanmin(Etot[:,:,0])) + ", Mean: " + str(np.nanmean(Etot[:,:,0]))
        print "Y: Max: " + str(np.nanmax(Etot[:,:,1])) + ", Min: " + str(np.nanmin(Etot[:,:,1])) + ", Mean: " + str(np.nanmean(Etot[:,:,1]))
        Q2 = plt.quiver( [275.0], [65.25], [-0.030], [0], color = 'k', scale = 0.75, width = 0.003) #banana for scale.
        
        cmap = plt.get_cmap('Greys')
        minval = 0.25
        maxval = 1
        n=100
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
        
        
        Q1 = plt.quiver( X, Y, Etot[:, :, 0], Etot[:, :, 1], MagnitudeForColor, cmap = new_cmap,  scale = 0.75, width = 0.003) #width = 0.003,
        plt.clim(0.0,30.0)
        plt.colorbar(Q1)
        
        
        #matplotlib.pyplot.scatter(mlonR[Iok], mlatR[Iok], s = 2)
    
    
    
    # clean the figure up a bit
    plt.xlim(np.nanmin(mlonArr), np.nanmax(mlonArr))
    plt.ylim(65, 69.5)
    plt.axis([np.nanmin(mlonArr), np.nanmax(mlonArr), 65, 69.5], fontsize = 22)
    
    
    
    # nice title containing relevant info
    plt.title(head_green['OBSSTART'], fontsize = 24)
    
    
    # Out file stuff
    out_str = str(head_green['OBSSTART'])
    out_str2 = out_str[0]+out_str[1]+out_str[3]+out_str[4]+out_str[6]+out_str[7]
    out_file = '%s_EfieldQuiver.jpg' % out_str2

 
    fig.savefig(dir_out + "/" + out_file)
    
    
    #############
    
    #fig = plt.figure(figsize = (14, 4))
    fig = plt.figure(figsize = (16, 12))
    
    xmin = np.nanmin(mlonArr)
    xmax = np.nanmax(mlonArr)
    
    ymin = np.nanmin(mlatArr)
    ymax = np.nanmax(mlatArr)
    
    
    plt.pcolormesh(mlonArr, mlatArr, data.T, vmin = 400, vmax = 700, cmap = 'ocean_r')
    #plt.colorbar(cmap = 'ocean_r')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    orig_cmap = matplotlib.cm.PuOr_r
    
    norm = matplotlib.colors.Normalize(vmin=divEGlobalMin, vmax=divEGlobalMax)

    shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(divEGlobalMax)/(np.absolute(divEGlobalMax) + np.absolute(divEGlobalMin))), name = 'shifted')
    
        

    #plt.contour(X, Y, divE, cmap = shifted_cmap, linewidths = 3.0)
    
    if timeStamp == '853':
        term = term1 + term2 + term3
        mlon = np.reshape(mlon, mlon.size, order = 'F')
        mlat = np.reshape(mlat, mlat.size, order = 'F')
        
        #Jmax = np.nanmax(Jparallel)
        #Jmin = np.nanmin(Jparallel)
        Jmax = np.nanmax(term)
        Jmin = np.nanmin(term)
        orig_cmap = matplotlib.cm.PuOr_r
        shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 1 - (np.absolute(Jmax)/(np.absolute(Jmax) + np.absolute(Jmin))), name = 'shifted')
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint = 0.333333333333, name = 'shifted')
        
        #plt.contour(mlon, mlat, Jparallel, cmap = shifted_cmap, vmin = np.nanmin(Jparallel), vmax = np.nanmax(Jparallel), linewidths = 3.0)
        plt.contour(mlon, mlat, term, cmap = shifted_cmap, vmin = Jmin, vmax = Jmax, linewidths = 3.0, levels = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        plt.colorbar()
        
         
    
    #plt.colorbar(cmap = shifted_cmap, format='%.0e') #norm = norm, 
  #  
        plt.xlim(mlon.min() - 0.5, mlon.max() + 0.5)
        plt.ylim(mlat.min() - 0.5, mlat.max() + 0.5)
        plt.axis([np.nanmin(mlon) - 0.5, np.nanmax(mlon) + 0.5, np.nanmin(mlat) - 0.5, np.nanmax(mlat) + 0.5])

    

    # clean the figure up a bit
    #plt.xlim(np.nanmin(mlonArr), np.nanmax(mlonArr))
    #plt.ylim(65, 69.5)
    
    
    # nice title containing relevant info
    #plt.title(head_green['FILTWAV']+'nm '+head_green['SITE']+' '+head_green['OBSDATE']+' '+head_green['OBSSTART'])
    
    
    out_str = str(head_green['OBSSTART'])
    
    out_str2 = out_str[0]+out_str[1]+out_str[3]+out_str[4]+out_str[6]+out_str[7]
    #out_file = '%s_FACContour_finerContours .jpg' % out_str2
    #out_file = '%s_divE.jpg' % out_str2
 
    plt.tight_layout()
    #fig.savefig(dir_out + "/" + out_file)
    
    plt.close('all')
    return None


def get_allsky_FA(file_in, dir_out, az_cal, el_cal): 
    
    #   Purpose
    #   This is the key routine
    #   The function of this routine is basically to align the all sky image with the 
    #   the field aligned beam from PFISR
    #   The images are in terms of pixels, so I use the where command to get the index
    #   that the beam corresponds to.  
    
    #   Input
    #   file_in =   including directory string
    #   dir_out =   out directory location.  I name the file below
    #   az_cal =    azimuth calibration fits file.  This is an all sky grid corresponding 
    #               to the azimuth angles for a given pixel
    #   el_cal=     Same as azimuth cal. A map of pixel locations with values of elevation
    
    #   Important Note: The az and el files are in geographic coordinates, but the 
    #   all sky images are in geomagnetic coordinates.  This was confirmed by Don. 
    
    #   Important Note #2: I tried to do this using the lon/lat files on the website
    #   when I plotted the beam lon/lat, it looked like nonsense so I abandoned it for the
    #   time being
    
    # read in data
    head_az, data_az = read_fits(az_cal) #('PKR_DASC_20110112_AZ_10deg.FITS')
    head_el, data_el = read_fits(el_cal)#('PKR_DASC_20110112_EL_10deg.FITS')
    head_green, data_green = read_fits(file_in)

    #   select only 5577 for the time being
    #   
    #   the data being read in should be transposed because of the IDL/python problem of
    #   rows being columns in python vs idl.
    #   the 180 degrees is to put it into the same frame
    #   This nominally aligns to the all sky movies found online.  It may not be perfect
    #   but it is good enough.
    data =rotate(np.transpose(data_green), angle=180)
    
    #convert the geographic az/el grid into geomagnetic coordinates
    # local field line at az = +22 and El = 77.5
    mag_az,mag_el =azel_geo2mag(data_az,data_el,22.,77.5) 
    
    #   rotate to put north at bottom of image and east to the right
    #   again, to be consistent with videos on website.
    mag_az = rotate(mag_az, angle=180)
    mag_el = rotate(mag_el,angle=180)





#   In case I want to put on a set of PFISR beams - here is how. 
#   beam_az =np.array([-154.3,-3.0,44.0,-154.3,10.3,28.8,-34.7,-5.9,50.2,75.0])#,20.5,-154.3])
#   beam_el = np.array([77.5,47.5,47.5,77.5,49.9,48.9,66.1,72.9,74.3,65.6])#,50.0,77.5])
# 
#   az_fa = -154.3
#   el_fa = 77.5
#   az_fa,el_fa = azel_geo2mag(-154.3,77.5,22.,77.5) 

#   I can plot those beams on the image if I want.
# for i,j in zip(mag_beam_az,mag_beam_el):
#   tmp_az,tmp_el = find_beam_pixels(mag_az,mag_el,i,j,0.25) 
#   print tmp_az,tmp_el
#   plt.plot(tmp_az,tmp_el, 'k.',mfc='none')


    # plot north and east az and el locations, so I know cardinal directions
    az_west = np.where((mag_az < -89.9) & (mag_az > -90.1))
    az_east = np.where((mag_az < 90.1) & (mag_az > 89.9))
    az_north = np.where((mag_az > 0) & (mag_az < 0.1))
    az_south = np.where(mag_az == 179)
    el_zenth = np.where((mag_el > 89.0) & (mag_el < 90.0) )
    
    
    
    # I had a hard time trying to plot the zenith beam, particularly since at 
    # field aligned the azimuth doesn't make much sense.
    # for the field aligned beam, I searched for points that were near 90 el and plotted those
    # that IS up B and it can be shown that the field aligned beam will reside in these limits
    
#   plt.plot(el_zenth[0][0],el_zenth[1][0], 'ko',mfc='none',mec='r')
    
    # clean the figure up a bit
#   plt.xlim(0,512)
#   plt.ylim(512,0)
    
    # nice title containing relevant info
#   plt.title(head_green['FILTWAV']+'nm '+head_green['SITE']+' '+head_green['OBSDATE']+' '+head_green['OBSSTART'])
    
    # Out file stuff
#   out_str = str(head_green['OBSSTART'])
#   out_str2 = out_str[0]+out_str[1]+out_str[3]+out_str[4]+out_str[6]+out_str[7]
#   out_file = '%s.png' % out_str2
    
    
    return data,head_green,el_zenth,az_west,az_east,az_north,az_south


# plot all sky field aligned only
def get_FA_pixel(file_in,az_cal,el_cal,az_fa,el_fa,el_min,el_max): 
    t1 = datetime.datetime.now()
    # read in data
    head_az, data_az = read_fits(az_cal) #('PKR_DASC_20110112_AZ_10deg.FITS')
    head_el, data_el = read_fits(el_cal)#('PKR_DASC_20110112_EL_10deg.FITS')
    head_green, data_green = read_fits(file_in)

    data =rotate(np.transpose(data_green), angle=180)

    mag_az,mag_el =azel_geo2mag(data_az,data_el,22.,77.5) 
    mag_az = rotate(mag_az, angle=180)
    mag_el = rotate(mag_el,angle=180)

#   az_fa = -154.3
#   el_fa = 77.5
    az_fa,el_fa = azel_geo2mag(az_fa,el_fa,22.,77.5) 

    el_zenth = np.where((mag_el > el_min) & (mag_el < el_max) )


    # now grab the time data.  
    obsdate = head_green['OBSDATE']
    obstime = head_green['OBSSTART']
    site = head_green['SITE']
    ccdtemp = head_green['CCDTEMP']
    exptime = head_green['EXPTIME']
    
    date_time = obsdate+' '+obstime
    t = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S.%f')
    t_UT = np.array([t.hour+(t.minute/60.)+(t.second/3600.)])
    

    
    pixVal = data_green[el_zenth]
    
    # make a dictionary, which is like a structure
    out = {'date_time':date_time, 
            'site':head_green['SITE'], 'ccdtemp':head_green['CCDTEMP'], 
            'timeUT':t_UT, 'pixVal':pixVal}
    t2 = datetime.datetime.now()
#   print t2-t1
    return out

# plot all sky field aligned only
def get_FA_pixel_428(file_in,az_cal,el_cal,az_fa,el_fa,el_min,el_max): 
    t1 = datetime.datetime.now()
    # read in data
    head_az, data_az = read_fits(az_cal) #('PKR_DASC_20110112_AZ_10deg.FITS')
    head_el, data_el = read_fits(el_cal)#('PKR_DASC_20110112_EL_10deg.FITS')
    head_green, data_green = read_fits(file_in)

    data =rotate(np.transpose(data_green), angle=180)

    mag_az,mag_el =azel_geo2mag(data_az,data_el,22.,77.5) 
    mag_az = rotate(mag_az, angle=180)
    mag_el = rotate(mag_el,angle=180)
    q1 = np.where((mag_el > 0) & (mag_el < 90) & (mag_az > 155) & (mag_az < 160))
#   az_fa = -154.3
#   el_fa = 77.5
    az_fa,el_fa = azel_geo2mag(az_fa,el_fa,22.,77.5) 

    el_zenth = np.where((mag_el > el_min) & (mag_el < el_max) )


    # now grab the time data.  
    obsdate = head_green['OBSDATE']
    obstime = head_green['OBSSTART']
    site = head_green['SITE']
    ccdtemp = head_green['CCDTEMP']
    exptime = head_green['EXPTIME']
    
    date_time = obsdate+' '+obstime
    t = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S.%f')
    t_UT = np.array([t.hour+(t.minute/60.)+(t.second/3600.)])
    
    # 10/21/2014 adding in calibration to Rayleighs from Don Hampton
#   print np.float(data_green[el_zenth])
    pixVal0 = data_green[el_zenth]
    pixVal1 = (np.float(data_green[el_zenth])-np.median(data_green[q1]))*(22./np.float(exptime))
    pixVal = np.absolute(pixVal1)
    print 'pixval', pixVal
    # make a dictionary, which is like a structure
    out = {'date_time':date_time, 
            'site':head_green['SITE'], 'ccdtemp':head_green['CCDTEMP'], 
            'timeUT':t_UT, 'pixVal0':pixVal0,'pixVal':pixVal}
    t2 = datetime.datetime.now()
#   print t2-t1
    return out

    
    
