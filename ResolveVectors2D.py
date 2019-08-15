# ResolveVectors2D.py

import tables
import numpy as np
import datetime as dt
from apexpy import Apex
import configparser

class ResolveVectors2D(object):
    def __init__(self):
        configfile = 'example_config.ini'
        self.read_config(configfile)

    def read_config(self, configfile):
        config = configparser.ConfigParser()
        config.read(configfile)
        self.__dict__.update(config.items('DEFAULT'))
        self.__dict__.update((name, eval(value)) for name, value in self.__dict__.items())

    def read_data(self):
        # read data from standard AMISR fit files
        with tables.open_file(self.datafile,'r') as infile:

            # time
            self.time = infile.get_node('/Time/UnixTime')[:]

            # site
            lat = infile.get_node('/Site/Latitude').read()
            lon = infile.get_node('/Site/Longitude').read()
            alt = infile.get_node('/Site/Altitude').read()
            self.site = np.array([lat, lon, alt/1000.])

            # define which beams to use (default is all)
            self.BeamCodes=infile.get_node('/BeamCodes')[:,0]
            try:
                bm_idx = np.array([i for i,b in enumerate(self.BeamCodes) if b in self.use_beams])
            except AttributeError:
                bm_idx = np.arange(0,len(self.BeamCodes))

            # geodetic location of each measurement
            self.alt = infile.get_node('/Geomag/Altitude')[bm_idx,:].flatten()
            self.lat = infile.get_node('/Geomag/Latitude')[bm_idx,:].flatten()
            self.lon = infile.get_node('/Geomag/Longitude')[bm_idx,:].flatten()

            # geodetic k vectors
            self.ke = infile.get_node('/Geomag/ke')[bm_idx,:].flatten()
            self.kn = infile.get_node('/Geomag/kn')[bm_idx,:].flatten()
            self.kz = infile.get_node('/Geomag/kz')[bm_idx,:].flatten()

            # line of sight velocity and error
            self.vlos = infile.get_node('/FittedParams/Fits')[:,bm_idx,:,0,3].reshape((len(self.time[:,0]),len(self.alt)))
            self.dvlos = infile.get_node('/FittedParams/Errors')[:,bm_idx,:,0,3].reshape((len(self.time[:,0]),len(self.alt)))

            # chi2 and fitcode (for filtering poor quality data)
            self.chi2 = infile.get_node('/FittedParams/FitInfo/chi2')[:,bm_idx,:].reshape((len(self.time[:,0]),len(self.alt)))
            self.fitcode = infile.get_node('/FittedParams/FitInfo/fitcode')[:,bm_idx,:].reshape((len(self.time[:,0]),len(self.alt)))

            # density (for filtering and ion upflow)
            self.ne = infile.get_node('/FittedParams/Ne')[:,bm_idx,:].reshape((len(self.time[:,0]),len(self.alt)))

            # temperature (for ion upflow)
            self.Te = infile.get_node('/FittedParams/Fits')[:,bm_idx,:,5,1].reshape((len(self.time[:,0]),len(self.alt)))
            Ts = infile.get_node('/FittedParams/Fits')[:,bm_idx,:,:5,1]
            frac = infile.get_node('/FittedParams/Fits')[:,bm_idx,:,:5,0]
            self.Ti = np.sum(Ts*frac,axis=-1).reshape((len(self.time[:,0]),len(self.alt)))

            # get up-B beam velocities for ion outflow correction
            if hasattr(self, 'upb_beamcode'):
                upB_idx = np.argwhere(self.BeamCodes==self.upb_beamcode).flatten()
                upB_alt = infile.get_node('/Geomag/Altitude')[upB_idx,:].flatten()
                upB_vlos = infile.get_node('/FittedParams/Fits')[:,upB_idx,:,0,3].reshape((len(self.time[:,0]),len(upB_alt)))
                upB_dvlos = infile.get_node('/FittedParams/Errors')[:,upB_idx,:,0,3].reshape((len(self.time[:,0]),len(upB_alt)))
                self.upB = {'alt':upB_alt, 'vlos':upB_vlos, 'dvlos':upB_dvlos}



    def filter_data(self):
        # filter and adjust data so it is appropriate for Bayesian reconstruction

        # add chirp to LoS velocity
        self.vlos = self.vlos + self.chirp

        # discard data with low density
        I = np.where((self.ne < self.nelim[0]) | (self.ne > self.nelim[1]))
        self.vlos[I] = np.nan
        self.dvlos[I] = np.nan

        # discard data outside of altitude range
        I = np.where((self.alt < self.altlim[0]*1000.) | (self.alt > self.altlim[1]*1000.))
        self.vlos[:,I] = np.nan
        self.dvlos[:,I] = np.nan

        # discard data with extremely high or extremely low chi2 values
        I = np.where((self.chi2 < self.chi2lim[0]) | (self.chi2 > self.chi2lim[1]))
        self.vlos[I] = np.nan
        self.dvlos[I] = np.nan

        # discard data with poor fitcode (fitcodes 1-4 denote solution found, anything else should not be used)
        I = np.where(~np.isin(self.fitcode, self.goodfitcode))
        self.vlos[I] = np.nan
        self.vlos[I] = np.nan


    def transform(self):
        # transform k vectors from geodetic to geomagnetic

        # find indices where nans will be removed and should be inserted in new arrays
        replace_nans = np.array([r-i for i,r in enumerate(np.argwhere(np.isnan(self.alt)).flatten())])

        glat = self.lat[np.isfinite(self.lat)]
        glon = self.lon[np.isfinite(self.lon)]
        galt = self.alt[np.isfinite(self.alt)]/1000.

        # intialize apex coordinates
        self.Apex = Apex(date=dt.datetime.utcfromtimestamp(self.time[0,0]))

        # find magnetic latitude and longitude
        mlat, mlon = self.Apex.geo2apex(glat, glon, galt)
        self.mlat = np.insert(mlat,replace_nans,np.nan)
        self.mlon = np.insert(mlon,replace_nans,np.nan)

        # apex basis vectors in geodetic coordinates [e n u]
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.Apex.basevectors_apex(glat, glon, galt)
        e1 = np.insert(e1,replace_nans,np.nan,axis=1)
        e2 = np.insert(e2,replace_nans,np.nan,axis=1)
        e3 = np.insert(e3,replace_nans,np.nan,axis=1)
        e = np.array([e1,e2,e3]).T

        # kvec in geodetic coordinates [e n u]
        kvec = np.array([self.ke, self.kn, self.kz]).T

        # find components of k for d1, d2, d3 base vectors (Laundal and Richmond, 2016 eqn. 60)
        self.A = np.einsum('ij,ijk->ik', kvec, e)

        # calculate scaling factor D, used for ion outflow correction (Richmond, 1995 eqn. 3.15)
        d1_cross_d2 = np.cross(d1.T,d2.T).T
        self.D = np.sqrt(np.sum(d1_cross_d2**2,axis=0))



    def ion_upflow_correction(self):

        # correct the los velocities for the entire array at each time
        for t in range(len(self.time)):
            if not hasattr(self, 'ionup'):
                continue
            elif self.ionup == 'UPB':
                # interpolate velocities from up B beam to all other measurements 
                vion, dvion = lin_interp(self.alt, self.upB['alt'], self.upB['vlos'][t], self.upB['dvlos'][t])
            elif self.ionup == 'EMP':
                # use empirical method to find ion upflow
                # NOTE: NOT DEVELOPED YET!!!
                vion, dvion = ion_upflow(self.Te, self.Ti, self.ne, self.alt)

            # LoS velocity correction to remove ion upflow
            self.vlos[t] = self.vlos[t] + self.A[:,2]/self.D*vion
            # corrected error in new LoS velocities
            self.dvlos[t] = np.sqrt(self.dvlos[t]**2 + self.A[:,2]**2/self.D**2*dvion**2)






def lin_interp(x, xp, fp, dfp):
    # Piecewise linear interpolation routine that returns interpolated values and their errors

    # find the indicies of xp that bound each value in x
    # Note: where x is out of range of xp, -1 is used as a place holder
    #   This provides a valid "dummy" index for the array calculations and can be used to identify values to nan in final output
    i = np.array([np.argwhere((xi>=xp[:-1]) & (xi<xp[1:])).flatten()[0] if ((xi>=np.nanmin(xp)) & (xi<np.nanmax(xp))) else -1 for xi in x])
    # calculate X
    X = (x-xp[i])/(xp[i+1]-xp[i])
    # calculate interpolated values
    f = (1-X)*fp[i] + X*fp[i+1]
    # calculate interpolation error
    df = np.sqrt((1-X)**2*dfp[i]**2 + X**2*dfp[i+1]**2)
    # replace out-of-range values with NaN
    f[i<0] = np.nan
    df[i<0] = np.nan

    return f, df



def main():
    rv = ResolveVectors2D()
    rv.read_data()
    rv.filter_data()
    rv.transform()
    rv.ion_upflow_correction()

if __name__=='__main__':
    main()