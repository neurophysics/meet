"""
Spatial Filters

Submodule of the Modular EEg Toolkit - MEET for Python.

This module implements some spatial filters such as CSP, CCA, CCAvReg,
bCSTP and QCA.

Author & Contact
----------------
Written by Gunnar Waterstraat
email: gunnar[dot]waterstraat[at]charite.de
"""
from __future__ import division
from . import _np
from . import _linalg
from . import _signal

def CSP(data1, data2, center=True):
    """
    Common Spatial Pattern (CSP)

    This takes the multivariate data in two conditions and finds the
    spatial filters which minimize the variance in condition 2, while
    simultaneously maximizing the variance in condition one.

    The algorithm uses Singular Value Decomposition since this is more
    stable than the Eigenvalue Decomposition of the Covariance Matrices

    The filters are scaled such, that Var(cond1) + Var(cond2) = 1. The
    eigenvalues give the variance in condition 1.
    The filters are in the columns of the filter matrix. Patterns can be
    obtained by inverting the filter matrix and are located in the rows
    of the inverted filter matrix.

    filter, eigval = CSP(data1, data2)
    filtered_data = filter.T.dot(data1)

    Input:
    ------
    -- data1 - 2d array - data in condition 1 (variables in rows,
          observations in columns)
    -- data2 - 2d array - data in condition 2 (variables in rows,
          observations in columns)
       The number of variables in data1 and data2 must be equal
    -- center - bool - if data should be centered, defaults to True

    Output:
    -------
    -- filter - where the individual filters are in the columns of the
          matrix (in order of decreasing eigenvalues)
    -- eigvals - eigenvalues in decreasing order
    """
    p1, n1 = data1.shape
    p2, n2 = data2.shape
    #remove mean
    if center:
        data1 = _signal.detrend(data1, axis=1, type='constant')
        data2 = _signal.detrend(data2, axis=1, type='constant')
    #normalize by signal length to have equal weights on all points
    data1 = data1
    data2 = data2 / (n2/float(n1))
    #whiten for data2
    W1, s = _linalg.svd(data2, full_matrices = False)[:2]
    #get rank
    rank1 = (s > (_np.max(s) * _np.max(data2.shape) *
        _np.finfo(data2.dtype).eps)).sum()
    # scale by singular values to get whitening matrix
    W = W1[:,:rank1] / s[:rank1][_np.newaxis]
    W2, s2 = _linalg.svd(W.T.dot(data1), full_matrices = False)[:2]
    rank2 = (s2 > (_np.max(s2) * _np.max([W.shape[1], data1.shape[1]]) *
        _np.finfo(data1.dtype).eps)).sum()
    both = W.dot(W2[:,:rank2])
    # scale by eigenvalue
    s2 = (s2[:rank2]**2) / (s2[:rank2]**2 +1)
    both = both * _np.sqrt((1-s2))[_np.newaxis] * _np.sqrt(n1)
    return both, s2

def CCA_data(X,Y):
    '''
    Cannonical Correlation Analysis - by a combination of QR
    decomposition and Singular Value Decomposition

    The filters are scaled to give unit variance in the components and
    are given in the order of decreasing canonical correlations.

    Patterns can be obtained by inverting the filter matrices.

    Inputs:
    -------
    X - shaped p1 x N - variables in rows, obervations in columns
    Y - shaped p2 x N - variables in rows, obervations in columns

    Outputs:
    --------
    a - filters for X (shape p1 x d), where d is min(rank(X), rank(Y)),
        each filter is in one column 
    b - filters for Y (shape p2 x d), where d is min(rank(X), rank(Y)),
        each filter is in one column 
    s - canonical correlations in non-increasing order, each
        corresponding to the respective colum in a and b
    '''
    import warnings
    ####################################################################
    ################# Checking input parameters ########################
    if not type(X) == _np.ndarray:
        raise TypeError('X must be numpy array')
    if not type(Y) == _np.ndarray:
        raise TypeError('Y must be numpy array')
    if not X.ndim == 2:
        raise ValueError('X must be a 2d numpy array - dimensionality' +
        'is wrong')
    if not Y.ndim == 2:
        raise ValueError('Y must be a 2d numpy array - dimensionality' +
        'is wrong')
    P1, M1 = X.shape
    P2, M2 = Y.shape
    if M1 != M2:
        raise ValueError('number of observations (columns) in X and Y' +
        'does not match.')
    if P1 <= 1:
        raise ValueError('number of Variables in X must be larger' +
        'than 1')
    if P2 <= 1:
        raise ValueError('number of Variables in Y must be larger' +
        'than 1')
    ####################################################################
    #remove mean
    X = _signal.detrend(X, axis=1, type='constant')
    Y = _signal.detrend(Y, axis=1, type='constant')
    # get QR decompositions
    qr_params = {'mode': 'economic', 'pivoting': True}
    Qx, Rx, permx = _linalg.qr(X.T, **qr_params)
    Qy, Ry, permy = _linalg.qr(Y.T, **qr_params)
    Rankx = _np.sum(_np.abs(_np.diag(Rx)) > (_np.finfo(Rx.dtype).eps *
        _np.max([M1,P1])))
    Ranky = _np.sum(_np.abs(_np.diag(Ry)) > (_np.finfo(Ry.dtype).eps *
        _np.max([M2,P2])))
    if Rankx == 0:
        raise ValueError('Rank of X is 0 -> it should contain at' +
        'least one non-constant column!')
    if Rankx < P1:
        warnings.warn(UserWarning('X is rank deficient!'))
        # take only relevant portions of Qx and Rx
        Qx = Qx[:,:Rankx]
        Rx = Rx[:Rankx,:Rankx]
    if Ranky == 0:
        raise ValueError('Rank of Y is 0 -> it should contain at' +
        'least one non-constant column!')
    if Ranky < P2:
        warnings.warn(UserWarning('Y is rank deficient!'))
        Qy = Qy[:,:Ranky]
        Ry = Ry[:Ranky,:Ranky]
    ####################################################################
    # get final number of dimensions
    d = _np.min([Rankx, Ranky])
    #get SVD
    U, s, Vh = _linalg.svd(Qx.T.dot(Qy), full_matrices=False,
            compute_uv=True)
    # get the filter matrices and normalize
    A_perm = _linalg.solve(Rx, U) * _np.sqrt(M1-1)
    B_perm = _linalg.solve(Ry, Vh.T) * _np.sqrt(M2-2)
    # initialize output arrays
    A = _np.zeros([P1,d], A_perm.dtype)
    B = _np.zeros([P2,d], B_perm.dtype)
    # revert order
    A[permx[:Rankx]] = A_perm
    B[permy[:Ranky]] = B_perm
    return A, B, s

def CCAvReg(trials):
    '''
    Canonical Correlation Average Regression

    Calculate the CCA between trials and their average across trials
    
    The filters are scaled to give unit variance in the components and
    are given in the order of decreasing canonical correlations.

    Patterns can be obtained by inverting the filter matrices.

    notably, all single trials can be filtered at once by:
    _np.tensordot(a, trials, axes=(0,0))

    the filtered average can be obtained as:
    _np.dot(b.T, trials.mean(-1))

    Input:
    ------
    -- trials - 3d numpy array - 1st dim: channels
                               - 2nd dim: epoch length
                               - 3rd dim: number of epochs

    Output:
    -------
    -- a - 2d numpy array (channel x channel) - spatial filters for
           single trials, each filter is in one column
    -- b - 2d numpy array (channel x channel) - spatial filters for
           average, each filter is in one column
    -- s - the canonical correlations between single trials and averages
    '''
    n_ch, n_dp, n_trials = trials.shape
    trials = _signal.detrend(trials,1, type='constant') # remove mean
    avg = trials.mean(2) # get average
    avg = _np.hstack([avg]*n_trials) # repeat the average to have the
    # same dimensionality as in trials
    # reshape the trials as 2d array
    trials = trials.reshape(n_ch, -1, order='F')
    # get the filters and eigenvalues
    a,b,s = CCA_data(trials, avg)
    return a, b, s

def bCSTP(trials1, trials2, num_iter=30, s_keep=2, t_keep=2,
        verbose=True):
    """
    bilinear Common Spatial-Temporal Patterns

    In each iteration the number of kept patterns is reduced by one,
    in the last 5 iterations the finally desired number of patterns is
    kept.
    The minimal number of iterations must therefor be > 10

    Filter matrices won't be square if the data was rank deficient,
    patterns can then be obtained by using the pseudo-inverse
    (numpy.linalg.pinv)

    Input:
    ------
    -- trials1 - 3d numpy array - 1st dim: channels
                                - 2nd dim: epoch length
                                - 3rd dim: number of epochs
       variance in condition 1 is maximized
    -- trials2 - 3d numpy array - 1st dim: channels
                                - 2nd dim: epoch length
                                - 3rd dim: number of epochs
       variance in condition 2 is minimized
    -- num_iter - int - number of iterations to do, defaults to 30
                        (minimum 10)
    -- s_keep - int - number of spatial patterns to keep, defaults to 2
    -- t_keep - int - number of temporal patterns to keep, defaults to 2
    -- verbose - bool - if number of iterations should be printed during
                        execution

    Output:
    -------
    -- W - list of 2d arrays - spatial filter matrix for each iteration,
           the final spatial filters are in W[-1]
    -- V - list of 2d arrays - temporal filter matrix for eachiteration,
                               the final temporal filters are in V[-1]
    -- s_eigvals - list of 1d array - spatial eigenvalues, 
                   the final spatial eigenvalues are in s_eigvals[-1]
    -- t_eigvals - list of 1d array - temporal eigenvalues, the final
                   temporal eigenvalues are in s_eigvals[-1]
    """
    n_ch1, n_dp1, n_trials1 = trials1.shape
    n_ch2, n_dp2, n_trials2 = trials2.shape
    #remove mean
    trials1 = _signal.detrend(trials1, axis=1, type='constant')
    trials2 = _signal.detrend(trials2, axis=1, type='constant')
    if n_ch1 != n_ch2:
        raise ValueError('Number of channels must be equal in trials1' +
                         'and trials2')
    if n_dp1 != n_dp2:
        raise ValueError('Epoch length must be equal in trials1 and' +
        'trials2')
    if num_iter < 10:
        raise ValueError('At least 10 iterations must be done.')
    t_keep = _np.min([_np.hstack([[int(t_keep)] * 5,
        _np.arange(6, num_iter+1, 1)])[::-1], [int(n_dp1)]*num_iter],0)
    s_keep = _np.min([_np.hstack([[int(s_keep)] * 5,
        _np.arange(6, num_iter+1, 1)])[::-1], [int(n_ch1)]*num_iter],0)
    # initialize output lists
    W = []
    V = [_np.identity(n_dp1, dtype=trials1.dtype)]
    t_eigvals = []
    s_eigvals = []
    for i in xrange(num_iter):
        if verbose:
            print 'Interation %d of %d total iterations.' % (i+1,
                    num_iter)
        ######## get spatial filter ########
        #filter temporally
        temp1 = _np.tensordot(V[-1][:,:t_keep[i]], trials1,
                axes = (0,1))
        temp2 = _np.tensordot(V[-1][:,:t_keep[i]], trials2,
                axes = (0,1))
        temp1 = temp1.swapaxes(0,1).reshape(n_ch1, -1, order='F')
        temp2 = temp2.swapaxes(0,1).reshape(n_ch2, -1, order='F')
        s_filt, s_val = CSP(temp1, temp2, center=False) # get CSP step
        # put into output lists
        W.append(s_filt)
        s_eigvals.append(s_val)
        ######## get temporal filter ########
        temp1 = _np.tensordot(W[-1][:,:s_keep[i]], trials1,
                axes = (0,0))
        temp2 = _np.tensordot(W[-1][:,:s_keep[i]], trials2,
                axes = (0,0))
        temp1 = temp1.swapaxes(0,1).reshape(n_dp1, -1, order='F')
        temp2 = temp2.swapaxes(0,1).reshape(n_dp2, -1, order='F')
        t_filt, t_val = CSP(temp1, temp2, center=False) # get CSP step
        # put into output lists
        V.append(t_filt)
        t_eigvals.append(t_val)
    #V was initialized as identity matrix, so start with the 2nd element
    return W, V[1:], s_eigvals, t_eigvals 
