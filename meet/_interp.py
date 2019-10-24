'''
hidden functions for interpolating EEG

Hidden submodule of the Modular EGg Toolkit - MEET for Python.

Author:
-------
Gunnar Waterstraat
gunnar[dot]waterstraat[at]charite.de
'''


from . import _np

def _get_akima_slopes(x,y, x_index):
    m1 = (y[x_index-1] - y[x_index-2]) / (x[x_index-1] - x[x_index-2])
    m2 = (y[x_index-0] - y[x_index-1]) / (x[x_index-0] - x[x_index-1])
    m3 = (y[x_index+1] - y[x_index+0]) / (x[x_index+1] - x[x_index+0])
    m4 = (y[x_index+2] - y[x_index+1]) / (x[x_index+2] - x[x_index+1])
    slopes = _np.zeros_like(m1)
    # indices for invalid output 
    i = _np.any([_np.all([m1==m2, m3==m4, m2!=m3],0), m2==m3],0)
    slopes[~i] = ((_np.abs(m4[~i] - m3[~i])*m2[~i] +
            _np.abs(m2[~i]-m1[~i])*m3[~i]) /
            (_np.abs(m4[~i]-m3[~i]) + _np.abs(m2[~i]-m1[~i])))
    return slopes

def _get_mchi_slopes(x,y,x_index):
    m_k   = (y[x_index+1] - y[x_index]) / (x[x_index+1] - x[x_index])
    m_k_1 = (y[x_index] - y[x_index-1]) / (x[x_index] - x[x_index-1])
    # get condition where slope is 0
    c = _np.any([m_k==0, m_k_1==0, _np.sign(m_k) != _np.sign(m_k_1)],0)
    ###
    h_k   = x[x_index[~c]+1] - x[x_index[~c]]
    h_k_1 = x[x_index[~c]] - x[x_index[~c]-1]
    w1 = 2*h_k + h_k_1
    w2 = h_k + 2*h_k_1
    whmean = 1. / (1./(w1+w2) * (w1/m_k[~c] + w2/m_k_1[~c]))
    m = _np.zeros(x_index.shape, whmean.dtype)
    m[~c] = whmean
    return m

def akima(x,y):
    x -= x.min()
    #the interpolated points are where x is discontinuous
    x1_index = _np.where(_np.diff(x) !=1)[0] # the points before
    x2_index = x1_index + 1 # the points after
    xi = [_np.arange(x[x1_index[k]]+1, x[x2_index[k]], 1) for k in
         range(len(x1_index))] # the points to be interpolated
    # get all relevant parameters
    t1 = _get_akima_slopes(x, y, x1_index)
    t2 = _get_akima_slopes(x, y, x2_index)
    x1 = x[x1_index]
    x2 = x[x2_index]
    y1 = y[x1_index]
    y2 = y[x2_index]
    p0 = y1
    p1 = t1
    p2 = (3*(y2 - y1)/(x2 - x1) - 2*t1 - t2) / (x2 - x1)
    p3 = (t1 + t2 -2*(y2 - y1)/(x2-x1)) / (x2-x1)**2
    yi = [p0[k] + p1[k]*(xi[k]-x1[k]) + p2[k]*(xi[k]-x1[k])**2 + p3[k]*
            (xi[k]-x1[k])**3 for k in range(len(x1_index))]
    return _np.hstack(yi)

def mchi(x,y):
    '''Monotone cubic hermite interpolation'''
    x -= x.min()
    #the interpolated points are where x is discontinuous
    x1_index = _np.where(_np.diff(x) !=1)[0] # the points before
    x2_index = x1_index + 1 # the points after
    xi = [_np.arange(x[x1_index[k]]+1, x[x2_index[k]], 1) for k in
         range(len(x1_index))] # the points to be interpolated
    # get all relevant parameters
    ###
    t1 = _get_mchi_slopes(x, y, x1_index)
    t2 = _get_mchi_slopes(x, y, x2_index)
    x1 = x[x1_index]
    x2 = x[x2_index]
    y1 = y[x1_index]
    y2 = y[x2_index]
    p0 = y1
    p1 = t1
    p2 = (3*(y2 - y1)/(x2 - x1) - 2*t1 - t2) / (x2 - x1)
    p3 = (t1 + t2 -2*(y2 - y1)/(x2-x1)) / (x2-x1)**2
    yi = [p0[k] + p1[k]*(xi[k]-x1[k]) + p2[k]*(xi[k]-x1[k])**2 + p3[k]*
            (xi[k]-x1[k])**3 for k in range(len(x1_index))]
    return _np.hstack(yi)
