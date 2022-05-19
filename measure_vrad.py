import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy.signal import correlate, correlation_lags
from scipy.constants import speed_of_light
import spectres

# filepath to directory containing fits and spectra folders
dir_path = '/home/tedsnowdon/Desktop'#'/home/tedsnowdon/salt.normalised'

def convert_degrees(str_in):
    # convert RA and DEC from xx:xx:xx string format to x.xxx... float format
    splitstr = str_in.split(':')
    return((float(splitstr[0])+float(splitstr[1])/60.+float(splitstr[2])/3600.)*u.deg)

def manual_barycorr(date, ra_in, dec_in):
    # Calculate barycentric correction for objects without a corresponding FITS file
    # requires date, and object RA/dec to be known
    salt = EarthLocation.of_site('SALT')
    sc = SkyCoord(ra=convert_degrees(ra_in), dec=convert_degrees(dec_in))
    barycorr = sc.radial_velocity_correction(obstime=Time(date), location=salt)
    return(barycorr.to(u.km/u.s))


def get_barycorr(obj_name, verbose=False):
    # Read coordinate and time data from fits header and return barycentric correction in km/s
    try:
        with fits.open(dir_path+'/fits/'+obj_name+'.fits') as hdul:
            date = hdul[0].header['DATE-OBS']
            site_lat = hdul[0].header['SITELAT']
            site_lon = hdul[0].header['SITELONG']
            site_alt = int(hdul[0].header['SITEELEV'])
            ra = hdul[0].header['RA']
            dec = hdul[0].header['DEC']
    
        if verbose:
            print('DATE: ', date)
            print('SITE LAT/LON/ALT: ', site_lat, site_lon, site_alt)
            print('RA/DEC: ', ra, dec)

        salt = EarthLocation.of_site('SALT') # Get site location automatically, requires internet connection!
        #salt = EarthLocation.from_geodetic(lat=site_lat*u.deg, lon=site_lon*u.deg, height=site_alt*u.m) # Manual alternative
        sc = SkyCoord(ra=convert_degrees(ra), dec=convert_degrees(dec))
        barycorr = sc.radial_velocity_correction(obstime=Time(date), location=salt)
        return(barycorr.to(u.km/u.s))
    except FileNotFoundError:
        print('NO MATCHING FITS FOUND FOR BARYCENTRIC CORRECTION')
        return(0.*(u.km/u.s))

def get_data(obj_name):
    # Read spectrum data from .sp2, convert wavelength to evenly-spaced log scale, rebin flux
    x = np.empty(0)
    y = np.empty(0)

    with open(dir_path+'/spectra/'+obj_name+'.sp2', 'r') as data_file:
        for line in data_file:
            split_line = line.split(' ')
            split_line[:] = [x for x in split_line if x] # removes the empty elements from the split string
            if len(split_line) == 2:
                x = np.append(x, np.log(float(split_line[0]))) # convert x into log wavelength units
                if split_line[1] != '-NaN\n':
                    y = np.append(y, float(split_line[1])-1.0) # subtract 1 from y to centre on zero
                else:
                    y = np.append(y,0)
    x_rebin = np.linspace(min(x), max(x), num=len(x)) # create constantly-spaced x axis
    y_rebin = spectres.spectres(x_rebin, x, y, fill=0, verbose=False) # rebin flux to match constantly-spaced x axis

    return [x_rebin, y_rebin]

def xcorr_specspec(obj_a, obj_b, plot=False):
    # Cross-correlate two spectra of the same object
    a = get_data(obj_a)
    print('--------------------')
    print('READING ', obj_a)
    print(len((a)[0]), 'DATA POINTS')
    a_vbary = get_barycorr(obj_a, verbose=True)
    print('BARYCENTRIC VELOCITY : ', a_vbary)

    b = get_data(obj_b)
    print('--------------------')
    print('READING ', obj_b)
    print(len((b)[0]), 'DATA POINTS')
    b_vbary = get_barycorr(obj_b, verbose=True)
    print('BARYCENTRIC VELOCITY : ', b_vbary)

    xcorr = correlate(a[1], b[1], mode='full') # perform cross correlation
    lags = correlation_lags(np.shape(a)[1], np.shape(b)[1], mode='full')

    parab_size = 2 # no. of points to take either side of xcorr peak, should be small
    parab_x = lags[np.argmax(xcorr)-parab_size:np.argmax(xcorr)+parab_size+1]
    parab_y = xcorr[np.argmax(xcorr)-parab_size:np.argmax(xcorr)+parab_size+1]

    parab = np.polyfit(parab_x, parab_y, 2) # fit curve to more precisely find peak of CCF

    if plot:
        parab_plot = []
        for i in parab_x:
            parab_plot.append((parab[0]*i**2)+(parab[1]*i)+parab[2])
        #plt.plot(lags, xcorr)
        #plt.plot(parab_x, parab_plot)
        #plt.title('cross correlation function')
        #plt.xlim([lags[np.argmax(xcorr)]-100, lags[np.argmax(xcorr)]+100])
        plt.plot(a[0], a[1])
        plt.plot(b[0], b[1]) 
        plt.xlim([8.4, 8.41])
        plt.show()

    peak_lag = -parab[1]/(2*parab[0])
    interval = np.mean(np.diff(a[0]))
    delta_loglam = interval*peak_lag
    delta_v = delta_loglam *(speed_of_light/1000)
    print('--------------------')
    print(obj_a, '*', obj_b)
    print('XCORR PEAK AT LAG = ', peak_lag)
    print('WAVELENGTH AXIS INTERVAL = ', interval)
    print('DELTA LOG(LAMBDA) = ', delta_loglam)
    print('DELTA V_RAD (KM/S) = ', delta_v)
    print('DELTA V_BAR (KM/S) = ', a_vbary-b_vbary)

xcorr_specspec('2019_BPS_VINCENTRENORM', 't3441v3244model_rebinned', plot=True)