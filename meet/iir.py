'''
IIR Filtering

Submodule of the Modular EEg Toolkit - MEET for Python.

So far only butterworth filtering is implemented

Author:
-------
Gunnar Waterstraat
gunnar[dot]waterstraat[at]charite.de
'''


from . import _np
from . import _signal

def butterworth(data, fp, fs, s_rate, gpass=3, gstop=8, axis=-1,
        zero_phase = True, return_param=False):
    '''
    Apply a butterworth filter to the data.

    fp is the passband frequency in Hz
    fs is the stopband frequency in Hz

    Example of Notations:
    ---------------------
    For fs = 10, fp = 15 a high-pass filter is applied with the
       transition band between 10-15 Hz
    For fs = 15, fp = 10 a low-pass filter is applied with the
       transition band between 10-15 Hz
    For fs = [10, 40], fp = [20,30] a bandpass-filter with the passband
       20-30 Hz and transitions between 10-20 Hz and 30-40 Hz is applied
    For fs = [20, 30], fp = [10,40] a bandstop-filter with the stopband
       20-30 Hz and transitions between 10-20 Hz and 30-40 Hz is applied

    Input:
    ------
    -- data - numpy array
    -- fp - float - pass-band frequencies
    -- fs - float - stop-band frequencies
    -- s_rate - float - the sampling rate in Hz
    -- gpass - float - maximal attenuation in the passband (in dB)
    -- gstop - float - minimal attenuation in the passband (in dB)
    -- axis - int -along which axis the filter is applied
    -- zero_phase - bool - if zero phase filter is applied by filtering
                    in both directions
    -- return_param - bool - if the order, b, and a shoud be returned

    Output:
    -------
    -- data - numpy array - the filtered data array
    if return_param:
       -- data
       -- ord - int - filter order (this must be doubled for the
                      zero_phase implementation)
       -- (b, a) - Numerator (`b`) and denominator (`a`) polynomials of
                   the IIR filter.
    '''
    # get filter parameters
    wp, ws, ftype = _preparefilter(fp, fs, s_rate)
    if zero_phase == True:
        gpass = gpass / 2. # half for filtering in both directions
        gstop = gstop / 2.
        filt = _signal.filtfilt
    else:
        filt = _signal.lfilter
    ord, wn = _signal.buttord(wp=wp, ws=ws, gpass=gpass, gstop=gstop)
    b,a = _signal.butter(N=ord, Wn = wn, btype=ftype)
    data = filt(b=b, a=a, x=data, axis=axis)
    if return_param:
        return data, ord, (b, a)
    return data

def _convertfreq(f, s_rate):
    '''
    Convert all frequencies to normalized frequency according to Nyquist
    frequency
    '''
    w = _np.array(f) / (s_rate/2.)
    return w

def _preparefilter(fp, fs, s_rate):
    '''
    Convert frequencies and get type of filter
    '''
    fp = _np.atleast_1d(fp)
    fs = _np.atleast_1d(fs)
    #get frequencies relative to Nyquist
    wp = _np.sort(_convertfreq(fp, s_rate))
    ws = _np.sort(_convertfreq(fs, s_rate))
    #get type of filter
    if len(wp) == 1:
        if ws < wp: ftype = 'highpass'
        if ws > wp: ftype = 'lowpass'
    if len(wp) == 2:
        if _np.all([ws[0] < wp[0], ws[1] > wp[1]],0): ftype = 'bandpass'
        if _np.all([ws[0] > wp[0], ws[1] < wp[1]],0): ftype = 'bandstop'
    return wp, ws, ftype
