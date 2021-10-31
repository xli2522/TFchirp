# Copyright (C) 2021 Xiyuan Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__version__ = '0.0.1'
__author__ = 'Xiyuan Li'

import numpy as np
import scipy.signal

def sTransform(ts, sample_rate, frange=[0, 800], frate = 1):
    '''Compute the S Transform
    Input:
                    ts                 (ndarray)            time series data
                    sample_rate        (int)                sample rate
                    frange             (list, optional)     frequency range (Hz)
                    frate              (int, optional)      frequency sampling rate
    Output:
                    amp                (ndarray)            spectrogram table
    Note:
                    amp                                     shape [frequency, time], lower -> higher
    '''

    length = len(ts)               
    Nfreq = [int(frange[0]*length/sample_rate), int(frange[1]*length/sample_rate)]      
    tsVal = np.copy(ts)            
    amp = np.zeros((int((Nfreq[1]-Nfreq[0])/frate)+1,length), dtype='c8')              
    tsFFT = np.fft.fft(tsVal)               
    vec = np.hstack((tsFFT, tsFFT))         

    amp[0] = np.fft.ifft(vec[0:length]*_window_normal(length, 0))           # set the lowest frequency row to small values => 'zero' frequency     
    for i in range(frate, (Nfreq[1]-Nfreq[0])+1, frate):                       
        amp[int(i/frate)] = np.fft.ifft(vec[Nfreq[0]+i:Nfreq[0]+i+length]*_window_normal(length, Nfreq[0]+i, factor=1))  

    return amp

def _window_normal(length, freq, factor = 1, elevated=True, elevation = 10e-8):
    '''Gaussian Window function w/ elevation
    Input: 
                    length              (int)               length of the Gaussian window
                    freq                (int)               frequency at which this window is to be applied to
                    factor              (int, float)        normalizing factor of the Gaussian; default set to 1
                    elevated            (bool, optional)    when True, add elevation to the Gaussian
                    elevation           (float, optional)   magnitude of the elevation     
    Output:
                    win                 (ndarray)           split gaussian window
    Note:
                    win                                     not your typical Gaussian => split + elevated (if True)
    '''
    gauss = scipy.signal.gaussian(length,std=(freq)/(2*np.pi*factor))
    if elevated:
        elevated_gauss = np.where(gauss<elevation, elevation,gauss)
        win = np.hstack((elevated_gauss,elevated_gauss))[length//2:length//2+length]
    else:
        win = np.hstack((gauss,gauss))[length//2:length//2+length]

    return win

def inverse_S(table, lowFreq = 0):
    '''Quick Inverse of S Transform Spectrogram
    Input:
                    table               (ndarray)           spectrogram table
                    lowFreq             (int, optional)     starting frequency
    Output:
                    ts_recovered        (ndarray)           recovered time series
    '''
    tablep = np.copy(table)                 
    length = tablep.shape[1]
    s_row = tablep[0]
    tsFFT_recovered = np.fft.fft(s_row)/_window_normal(length, lowFreq)

    ts_recovered = np.fft.ifft(tsFFT_recovered)

    return ts_recovered