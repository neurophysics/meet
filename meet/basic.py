'''
Basic functions for reading binaries and EEG manipulation

Submodule of the Modular EEg Toolkit - MEET for Python.

Author:
-------
Gunnar Waterstraat
gunnar[dot]waterstraat[at]charite.de
'''

from __future__ import division
from . import _np
from . import _path
from . import _packdir

def readBinary(fname,num_channels,channels='all',readnum_dp='all',
               data_type='float4', buffermem=512):
    '''
    Read EEG from a binary file and output as numpy array.
    The binary of a signal with k channels and n datapoints must be
    of the type:

            t0   t1 ... ... tn-1
            ----------------------------
    ch_0  | 0    k      ... (n-1)*k
    ch_1  | 1    k+1    ... (n-1)*k+1
    ...   | ...  ...    ... ...
    ch_k-1| k-1  2*k-1  ... n*k-1

    The endianess of the runtime system is used.

    Input:
    ------
    -- fname - (str) - input file name
    -- num_channels - int - total number of channels in the file
    -- channels - numpy array OR 'all' - iterable of channels to read
                  (starting with 0) if 'all', all channels are read
    -- readnum_dp - int OR 'all' - number of datapints to read
    -- data_type - str - any of 'int2', 'int4', 'int8', 'float4',
                         'float8', 'float16' where the digit determins
                         the number of bytes (= 8 bits) for each element
    -- buffermem - float  - number of buffer to us in MegaBytes

    Output:
    -------
    -- data - numpy array - data shaped k x n where k is number of
                            channels and n is number of datapoints

    Example:
    --------
    >>> readBinary(_path.join(_path.join(_packdir, 'test_data'), \
            'sample.dat'), 2, data_type='int8')
    array([[0, 2, 4, 6, 8],
           [1, 3, 5, 7, 9]])
    '''
    from os.path import getsize
    if data_type == 'float4':
        bytenum = 4
        id = 'f'
    elif data_type == 'float8':
        bytenum = 8
        id = 'f'
    elif data_type == 'float16':
        bytenum = 16
        id = 'f'
    elif data_type == 'int2':
        bytenum = 2
        id = 'i'
    elif data_type == 'int4':
        bytenum = 4
        id = 'i'
    elif data_type == 'int8':
        bytenum = 8
        id = 'i'
    else: raise ValueError('Data type not recognized.')
    # get the length of the dataset
    filesize = getsize(fname)
    #get number of datapoints
    data_num = filesize // bytenum // num_channels
    if channels =='all':
        channels = _np.arange(num_channels)
    if readnum_dp =='all':
        readnum_dp = data_num
    fd = open(fname,'rb') #open file
    # get number of batches to read dataset
    bytequot = int(int(buffermem * 1024**2 / bytenum) -
               (buffermem * 1024**2 / bytenum) % num_channels)
    batchnum = int(_np.ceil(num_channels*readnum_dp / float(bytequot)))
    num_dp_per_batch = bytequot / num_channels
    readnum_ch = len(channels)
    data = _np.empty([readnum_ch,readnum_dp],dtype='<'+id+str(bytenum))
    if (((num_channels*readnum_dp) % bytequot == 0) or
         (num_channels*readnum_dp) < bytequot):
        #if the dataset can be read in complete batches
        for i in xrange(batchnum):
            #read all channels
            data_temp=_np.fromfile(
              file = fd,
              count = bytequot,
              dtype= '<'+id+str(bytenum)).reshape([num_channels,-1],
                      order='f')
            #assign the wanted chanels
            data[:,i*num_dp_per_batch:i*num_dp_per_batch +
                    num_dp_per_batch] = data_temp[channels,:]
        fd.close()
    else:
        #if partial batches are needed at the end
        for i in xrange(batchnum-1):
            #read all channels
            data_temp=_np.fromfile(
              file = fd,
              count = bytequot,
              dtype= '<'+id+str(bytenum)).reshape([num_channels,-1],
                      order='f')
            #assign the wanted chanels
            data[:,i*num_dp_per_batch:i*num_dp_per_batch +
                    num_dp_per_batch] = data_temp[channels,:]
        #read all channels
        data_temp=_np.fromfile(
          file=fd,
          count = (num_channels*readnum_dp) % bytequot,
          dtype= '<'+id+str(bytenum)).reshape([num_channels,-1],
                  order='f')
        #assign to wanted chanels
        data[:,(batchnum-1)*num_dp_per_batch:] = data_temp[channels,:]
        fd.close()
    return data

def interpolateEEG(data, markers, win, interpolate_type='mchs'):
    """
    Interpolates segemnets in the data

    Input:
    ------
    -- data - one or two dimensional array
              1st dimension: channels (can be ommited if single channel)
              2nd dimension: datapoints
    -- markers - marker positions arranged in 1d array
    -- win - iterable of len 2 - determining the window in datapoints to
             be interpolated (win[0] is in, win[1] is out of the window)
    -- interpolate_type: ['linear', 'mchs', 'akima'] - linear or
                         Monotone Cubic Hermite Spline
       or Akima interpolation

    Output:
    -------
    interpolated dataset

    Examples:
    --------
    >>> data = _np.arange(20, dtype=float).reshape(2,-1)
    >>> interpolateEEG(data, [5], [-1,2], 'linear')
    array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
           [ 10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.]])
    >>> interpolateEEG(data, [5], [-1,2], 'mchs')
    array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
           [ 10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.]])
    >>> interpolateEEG(data, [5], [-1,2], 'akima')
    array([[  0.   ,   1.   ,   2.   ,   3.   ,   3.625,   5.   ,   6.375,
              7.   ,   8.   ,   9.   ],
           [ 10.   ,  11.   ,  12.   ,  13.   ,  13.625,  15.   ,  16.375,
             17.   ,  18.   ,  19.   ]])
    """
    interpolpts = [_np.arange(m+win[0], m+win[1],1) for m in markers] 
    interpolpts = _np.sort(_np.ravel(interpolpts))
    have_indices = _np.ones(data.shape[1],bool)
    have_indices[interpolpts] = False
    x = _np.arange(data.shape[1])[have_indices]
    if interpolate_type == 'linear':
        from scipy.interpolate import interp1d as interp
        f = interp(x, data[:,have_indices], axis=-1)
        data[:,interpolpts] = f(interpolpts)
    elif interpolate_type in ['pchip', 'akima']:
        if interpolate_type == 'akima':
            from _interp import akima as interp
        elif interpolate_type == 'mchs':
            from _interp import mchi as interp
        if data.ndim == 1:
            data[interpolpts] = interp(x, data[have_indices])
        elif data.ndim == 2:
            for ch in xrange(data.shape[0]):
                data[ch, interpolpts] = interp(x,
                        data[ch, have_indices])
    return data


def epochEEG(data, marker, win):
    """
    Arange the dataset into trials (=epochs) according to the marker and
    window.

    markers and the window borders are sorted in ascending order.

    Input:
    ------
    -- data - numpy array - 1st dim channels (can be ommited if single
                                              channel)
                            2nd dim datapoints
    -- marker - iterable - the marker
    -- win - iterable of len 2 - determing the start and end of epchos
             in dp (win[0] is in, win[1] is out of the window)

    Output:
    -------
    -- epochs - numpy array - dimension one more then data input
                            - 1st dim: channel (might be ommited - see
                                                above)
                            - 2nd dim: epoch length = win[1] - win[0]
                            - 3rd dim: number of epochs

    Example:
    --------
    >>> data = _np.arange(20, dtype=float).reshape(2,-1)
    >>> epochEEG(data, [3,5,7], [-2,2])
    array([[[  1.,   3.,   5.],
            [  2.,   4.,   6.],
            [  3.,   5.,   7.],
            [  4.,   6.,   8.]],
    <BLANKLINE>
           [[ 11.,  13.,  15.],
            [ 12.,  14.,  16.],
            [ 13.,  15.,  17.],
            [ 14.,  16.,  18.]]])
    """
    win = _np.sort(win)
    marker = _np.sort(marker)
    p,n = data.shape
    #omit marker that would allow only for incomplete windows
    if (marker[0] + win[0]) < 0:
        marker = marker[marker+win[0] > 0]
        print ('Warning: Marker had to be ommited since for some' +
               'marker + win[0] < 0')
    if (marker[-1] + win[1]) >= n:
        marker = marker[marker+win[1] < n]
        print ('Warning: Marker had to be ommited since for some' +
               'marker + win[1] >= len(data)')
    indices = _np.array([_np.arange(m + win[0], m+win[1], 1)
                         for m in marker])
    return data.T[indices].T

def calculateRMS(data, axis=-1):
    """
    Calculate rms value of the input data along the indicated axis
    
    Input:
    ------
    -- data - numpy array - input data
    -- axis - int - axis along which the rms is calculated; if None, the
                    flattened array is used
    Output:
    -------
    -- rms value along the indicated axis

    Example:
    --------
    >>> data = _np.arange(20, dtype=float).reshape(2,-1)
    >>> calculateRMS(data, None)
    11.113055385446435
    >>> calculateRMS(data, 0)
    array([  7.07106781,   7.81024968,   8.60232527,   9.43398113,
            10.29563014,  11.18033989,  12.08304597,  13.        ,
            13.92838828,  14.86606875])
    >>> calculateRMS(data, 1)
    array([  5.33853913,  14.7817455 ])
    """
    if axis == None:
        return _np.sqrt((_np.ravel(data)**2).mean())
    else:
        data = data.swapaxes(0,axis) # now calculate everything along
                                     # axis 0
        return _np.sqrt((data**2).mean(0))

def getMarker(marker, width=50, mindist=100):
    """
    Gets position of markers from the trigger channel
    GetMarkerPosFromData(marker)

    input:
    -- marker - one-dimensional array with trigger channel - each
                impulse or zero crossing is treated a marker
    --width - int - calculates the local mean in window of size width
                    - defaults to 50
    --mindist - int - minimal distance between triggers in dp
                      - defaults to 100
    output:
    -- marker - one-dimensional array containing trigger positions

    Example:
    --------
    >>> x = _np.ones(1000)
    >>> x[200:400] = -1
    >>> x[600:800] = -1
    >>> getMarker(x)
    array([ 400,  600,  800, 1000])
    """ 

    # normalize by local median and mad
    # add some random noise to prevent the median from being zero later
    marker = (marker +
            _np.random.random(len(marker)) * marker.ptp() * 1E-5)
    mean = _np.convolve(marker, _np.ones(width)/float(width),
                        mode='same') # moving average
    # median of deviation from local mean
    mad = _np.median(_np.abs(marker - mean))
    # weight the local deviation to average deviation and find crossings
    # above the 20-fold mad
    marker = _np.abs(marker - mean) / mad - 20 
    results = _np.where(_np.all([marker[:-1] * marker[1:] < 0,
              marker[:-1] < 0], axis=0))[0] # find zero crossings
    too_close = True
    while too_close:
        results = (results[::-1][_np.diff(results)[::-1] >
                   mindist])[::-1]
        if _np.all(_np.diff(results) > mindist): too_close = False
    return results + int(width/2.)
