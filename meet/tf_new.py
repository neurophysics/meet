'''
S transform (time-frequency transformation)

Submodule of the Modular EEg Toolkit - MEET for Python.

the 'standard' S transform:
----------------------------
Stockwell, Robert Glenn, Lalu Mansinha, and R. P. Lowe. "Localization of
the complex spectrum: the S transform." Signal Processing, 
IEEE Transactions on 44.4 (1996): 998-1001.

as well as

the fast dyadic S transform:
---------------------------------------
Brown, Robert A., M. Louis Lauzon, and Richard Frayne. "A general
description of linear time-frequency transforms and formulation of a
fast, invertible transform that samples the continuous S-transform
spectrum nonredundantly." Signal Processing, IEEE Transactions on 58.1
(2010): 281-290.

Author:
-------
Gunnar Waterstraat
gunnar[dot]waterstraat[at]charite.de

Example for a standard S transform:
-----------------------------------
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> data = np.random.random(1000) # generate a random 1000 dp signal
>>> tf_coords, S = gft(data, sampling='full')
>>> S = S.reshape(-1, data.shape[0]) # for the full transform no \
        interpolation is needed -> just reshape to get a regular grid
>>> plt.imshow(_np.abs(S), aspect='equal', origin='lower') #plot the \
        result


Example for a dyadic S transform:
-----------------------------------
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> data = np.random.random(1000) # generate a random 1000 dp signal
>>> tf_coords, S = gft(data, sampling='dyadic')
>>> t, f, S_interp = interpolate_gft(tf_coords, S, (data.shape[0]//2, \
        data.shape[0]), data.shape[0], kindf = 'nearest', \
        kindt = 'nearest')
>>> plt.imshow(_np.abs(S_interp), aspect='equal', origin='lower', \
        extent=[t[0], t[-1], f[0], f[-1]]) #plot the result
'''

from __future__ import division 
from . import _np
from . import _signal
from . import _sci

# Sampling Schemes
def _dyadic(N):
    '''
    Get centre frequencies and corresponding start and stop
    frequencies
    
    Input:
    ------
    -- N - length of signal in dp

    Output:
    -------
    -- y
    -- x
    '''
    y =  _np.array([0] + list(2 ** _np.arange(int(_np.ceil(_np.log2(N)))-1)))
    x = _np.where(y<2, 0, y//2)
    return x,y

def _full_sampling(N):
    y = _np.arange(N//2)
    x = _np.ones_like(y) * N/2.
    return x,y

#FT of Window
def _gaussian_ft(N, f):
    x = _np.fft.fftfreq(N, d = 1./N)
    win = _np.exp(x**2 * -1*2*_np.pi**2/float(f**2))
    return win

def gft(sig, window='gaussian', axis=-1, sampling = 'full', full=False,
        hanning=True):
    '''
    Calculate the discrete general fourier family transform
    
    Inputs:
    -------
    -- sig - signal of interest - 1 or 2 dimensional
    -- window - window function, standard is gaussian
    -- axis - axis along which transform should be performed
    -- sampling - 'full'|'dyadic' - should the S-domain be sampled
                  completely, i.e. redundant or dyadic 
    -- full - True|False - return also negative frequencies, standard
              is False
    -- hanning - True|False - Apply a hanning window to 5% first and
                 last datapoints
    
    Outputs:
    --------
    -- Coords - (frequency, time) Coordinates of each point in S
    -- S - complex S transform array
       if sampling is dyadic interpolate_gft should be used to
       interpolate the result onto a regular grid
       if sampling is full, this can be easily done by reshaping each S
       transform to the shape (-1, sig.shape[axis]) 
    '''
    #Step0: Preparational steps
    sig = sig.swapaxes(axis, 0) # Make the relevant axis axis 0
    sig_mean = sig.mean(axis=0)
    sig = _signal.detrend(sig, axis=0, type='constant')
    N = sig.shape[0]
    if hanning == True:
        hann_N = int(0.1*N)
        if hann_N % 2 != 0:
            hann_N += 1
        hann = _signal.hanning(hann_N)
        sig[:hann_N/2] = (sig[:hann_N/2].T * hann[:hann_N/2]).T
        sig[-hann_N/2:] = (sig[-hann_N/2:].T * hann[-hann_N/2:]).T
    if (not full in [True, 'True', 1, '1']) & _np.all(_np.isreal(sig)):
        sig = _signal.hilbert(sig, axis=0)
    if window == 'gaussian': window = _gaussian_ft
    if type(sampling) == str:
        if sampling == 'dyadic': sampling = _dyadic
        if sampling == 'full': sampling = _full_sampling
    #Step1: FT of Signal
    sig = _np.fft.fft(sig, axis=0) # get fft of signal
    #Step2: Sampling Scheme
    x,y = sampling(N)
    for k in xrange(len(y)):
        if y[k] != 0:
            sig_r = _np.roll(sig, -y[k], axis=0)
            win = window(N, y[k])
            if x[k] == 0:
                indices = _np.array([0])
            else:
                if x[k] % 1 == 0:
                    indices = _np.arange(-int(x[k]), int(x[k]), 1)
                else:
                    indices = _np.arange(-int(x[k]), int(x[k])+1, 1)
            alpha = (sig_r[indices].T * win[indices]).T
            s = _np.fft.ifft(_np.fft.ifftshift(alpha, axes=0), axis=0)
        else:
            if x[k] == 0:
                s = _np.asanyarray(sig_mean)[_np.newaxis]
            else:
                if x[k] % 1 == 0:
                    s = _np.ones([2*x[k]]+list(sig.shape[1:]),
                        dtype = sig_mean.dtype) * (1./(2*x[k])) * sig_mean[_np.newaxis]
                else:
                    s = _np.ones([2*int(x[k]) + 1]+list(sig.shape[1:]),
                        dtype = sig_mean.dtype) * (1./(2*x[k]+1)) * sig_mean[_np.newaxis]
        #t_step = N/float(len(s)) # find step size in time
        #t_now = _np.arange(0,N,t_step)  + t_step/2.
        t_now = _np.linspace(0,N,len(s),endpoint=False)
        # add half a step, so that it will always be in the center
        t_now = t_now + (N - t_now[-1])/2.
        #
        f_now = y[k] * _np.ones_like(t_now)
        coords = _np.vstack([f_now, t_now])
        s = s.T
        if k == 0:
            Coords = coords
            S = s
        else:
            Coords = _np.hstack([Coords,coords])
            S = _np.concatenate([S,s], axis=-1)
    if sig.ndim > 1:
        S = S.swapaxes(0,axis)
    return Coords, S

def interpolate_gft(Coords, S, IM_shape, data_len, kindf = 'nearest',
        kindt = 'nearest'):
    '''
    Interpolate the result of a gft-transform - standard is nearest
    neighbor interpolation
    
    Input:
    ------
    -- Coords, Coords as output by the gft command
    -- S - A gft result list
    -- IM_shape - requested shape of the interpolated matrix
           1st axis frequency
           2nd axis time
    -- data_len - length of inital data
    -- kindf - interpolation method along frequency axis - any of
               ['nearest', 'linear']
    -- kindt - interpolation method along time axis - any of
               ['nearest', 'linear']
    
    Output:
    ------
    -- wanted times - time point array
    -- wanted freqs - frequency array
    -- IM - interpolated matrix
    '''
    ######################################
    # Initialize matrix
    IM = _np.zeros(IM_shape, S.dtype)
    # find unique frequency values
    f, f_indices = _np.unique(Coords[0], return_index=True)
    f = f[_np.argsort(f_indices)]
    f_order = _np.argsort(f)
    # find indices where frequency changes
    f_indices = _np.where(_np.diff(Coords[0]) != 0)[0] + 1
    f_indices = _np.hstack([0, f_indices, len(S)])
    # Initialize temporary interpolation matrix
    IM_temp = _np.empty([len(f), IM_shape[1]], S.dtype)
    wanted_times = _np.linspace(0,data_len, IM_shape[1])
    if f.min() >= 0:
        wanted_freqs = _np.linspace(0,int(data_len / 2), IM_shape[0])
    else:
        # if negative frequencies are included interpolate for complete
        # spectrum
        wanted_freqs = _np.linspace(-int(data_len / 2),
                int(data_len / 2), IM_shape[0])
    for k in xrange(len(f)):
        x = Coords[1,f_indices[k]:f_indices[k+1]]
        y = S[f_indices[k]:f_indices[k+1]]
        if x[0] !=0:
            x = _np.hstack([0, x])
            y = _np.hstack([y[0] ,y])
        if x[-1] != data_len:    
            x = _np.hstack([x, data_len])
            y = _np.hstack([y, y[-1]])
        interfunc = _sci.interp1d( x=x, y=y, kind=kindt)
        IM_temp[k] = interfunc(wanted_times)
        #IM_temp[k] = S[f_indices[k]:f_indices[k+1]]
    for k in xrange(IM_shape[1]):
        interfunc = _sci.interp1d(x = f[f_order],
                y = IM_temp[f_order,k], kind = kindf )
        IM[:,k] = interfunc(wanted_freqs.clip(f.min(),f.max()))
    return wanted_times, wanted_freqs, IM
