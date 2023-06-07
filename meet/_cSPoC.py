"""
Implements canonical Source Power Correlation analysis (cSPoC)

    Reference:
    ----------
    Dahne, S., et al., Finding brain oscillations with power dependencies
    in neuroimaging data, NeuroImage (2014),
    http://dx.doi.org/10.1016/j.neuroimage.2014.03.075

Hidden submodule of the Modular EEg Toolkit - MEET for Python.
(Imported by spatfilt module)

This module implements some spatial filters such as CSP, CCA, CCAvReg,
bCSTP and QCA.
"""


from . import _np #numpy
from . import _signal #scipy.signal
from . import _linalg #scipy.linalg
from scipy.optimize import fmin_l_bfgs_b as _minimize

def pattern_from_filter(filter, X):
    """
    Get a an activation pattern from the filter matrix
    and the input data X.

    Input:
    ------
    filter - is a p (x k) numpy array, where p is the dimensionality of the
             data and k is the number of filters (if k == 1, the array may
             be 1-dimensional)
    X - is an p x N (x tr) numpy array, where p is the dimensionality of
        the data, N is the number of datapoints (per trial, and tr is the
        number of trials)
        If X is 3d, the patterns are computed by collapsing the last 2 axes

    Output:
    -------
    pattern - is a (k x) p numpy array of activation patterns, where p is
              the dimensionality of the data and k is the number of filters
              If the input filter array is 1d so will be this array
    """
    assert isinstance(filter, _np.ndarray), "filter must be a numpy array"
    assert filter.ndim <=2, "filter dimensionality must be 1 or 2"
    assert isinstance(X, _np.ndarray), "X must be a numpy array"
    assert X.ndim in [2,3], "dimensionality of X must be 2 or 3"
    assert X.shape[0] == filter.shape[0], 'First dimension of filter' + \
                                          'and X must agree!'
    b = _np.cov(X.reshape(X.shape[0], -1)).dot(filter)
    a = _np.cov(filter.T.dot(X.reshape(X.shape[0], -1)))
    pat = _linalg.solve(_np.atleast_2d(a), _np.atleast_2d(b.T))
    if filter.ndim == 1: pat = pat[0]
    return pat

def cSPoC(X, Y, opt='max', num=1, log=True, bestof=15, x_ind=None, y_ind=None):
    """
    canonical Soure Power Correlation analysis (cSPoC)

    For the datasets X and Y, find a pair of linear filters wx and wy, such
    that the correlation of the amplitude envelopes wx.T.dot(X) and
    wy.T.dot(Y) is maximized.

    Reference:
    ----------
    Dahne, S., et al., Finding brain oscillations with power dependencies
    in neuroimaging data, NeuroImage (2014),
    http://dx.doi.org/10.1016/j.neuroimage.2014.03.075

    Notes:
    ------
    Datasets X and Y can be either 2d numpy arrays of shape
    (channels x datapoints) or 3d array of shape
    (channels x datapoints x trials).
    For 3d arrays the average envelope in each trial is calculated if x_ind
    (or y_ind, respectively) is None. If they are set, the difference of
    the instantaneous amplitude envelope at x_ind/y_ind and the average
    envelope is calculated for each trial.
    If log == True, then the log transform is taken before the average
    inside the trial

    If X and/or Y are of complex type, it is assumed that these are the
    analytic representations of X and Y, i.e., the hilbert transform was
    applied before.

    The filters are in the columns of the filter matrices Wx and Wy,
    for 2d input the data can be filtered as:

    np.dot(Wx.T, X)

    for 3d input:

    np.tensordot(Wx, X, axes=(0,0))

    Input:
    ------
    -- X numpy array - the first dataset of shape px x N (x tr), where px
                       is the number of sensors, N the number of data-
                       points, tr the number of trials. If X is of complex
                       type it is assumed that this already is the
                       analytic representation of X, i.e. the hilbert
                       transform already was applied.
    -- Y is the second dataset of shape py x N (x tr)
    -- opt {'max', 'min'} - determines whether the correlation coefficient
                            should be maximized - seeks for positive
                            correlations ('max', default);
                            or minimized - seeks for anti-correlations
                            ('min')
    -- num int > 0 - determine the number of filter-pairs that will be
                     derived. This depends also on the ranks of X and Y,
                     if X and Y are 2d the number of filter pairs will be:
                     min([num, rank(X), rank(Y)]). If X and/or Y are 3d the
                     array is flattened into a 2d array before calculating
                     the rank
    -- log {True, False} - compute the correlation between the log-
                           transformed envelopes, if datasets come in
                           epochs, then the log is taken before averaging
                           inside the epochs, defaults to True
    -- bestof int > 0 - the number of restarts for the optimization of the
                        individual filter pairs. The best filter over all
                        these restarts with random initializations is
                        chosen, defaults to 15.
    -- x_ind int - the time index (-X.shape[1] <= x_ind < X.shape[1]) where
                   the difference of the instantaneous envelope and the
                   average envelope is determined for X
    -- y_ind int - the time index (-Y.shape[1] <= y_ind < Y.shape[1]) where
                   the difference of the instantaneous envelope and the
                   average envelope is determined for Y

    Output:
    -------
    corr - numpy array - the canonical correlations of the amplitude
                         envelopes for each filter
    Wx - numpy array - the filters for X, each filter is in a column of Wx
                       (if num==1: Wx is 1d)
    Wy - numpy array - the filters for Y, each filter is in a column of Wy
                       (if num==1: Wy is 1d)
    """
    #check input
    assert isinstance(X, _np.ndarray), "X must be numpy array"
    assert (X.ndim ==2 or X.ndim==3), "X must be 2D or 3D numpy array"
    assert isinstance(Y, _np.ndarray), "X must be numpy array"
    assert (Y.ndim ==2 or Y.ndim==3), "Y must be 2D or 3D numpy array"
    assert X.shape[-1] == Y.shape[-1], "Size of last dimension in X and" \
                                     + " Y must be equal"
    assert opt in ['max', 'min'], "\"opt\" must be \"max\" or \"min\""
    assert isinstance(num, int), "\"num\" must be integer > 0"
    assert num > 0, "\"num\" must be integer > 0"
    assert log in [True, False, 0, 1], "\"log\" must be a boolean (True " \
                                     + "or False)"
    assert isinstance(bestof, int), "\"bestof\" must be integer > 0"
    assert bestof > 0, "\"bestof\" must be integer > 0"
    if x_ind != None:
        assert X.ndim == 3, "If x_ind is set, X must be 3d array!"
        assert isinstance(x_ind, int), "x_ind must be integer!"
        assert ((x_ind >= -X.shape[1]) and
                (x_ind < X.shape[1])), "x_ind must match the range of " +\
                                       "X.shape[1]"
    if y_ind != None:
        assert Y.ndim == 3, "If y_ind is set, Y must be 3d array!"
        assert isinstance(y_ind, int), "y_ind must be integer!"
        assert ((y_ind >= -Y.shape[1]) and
                (y_ind < Y.shape[1])), "y_ind must match the range of " +\
                                       "Y.shape[1]"
    # get whitening transformations
    # for X
    Whx, sx = _linalg.svd(X.reshape(X.shape[0],-1).real, full_matrices=False)[:2]
    #get rank
    px = (sx > (_np.max(sx) * _np.max([X.shape[0],_np.prod(X.shape[1:])]) *
          _np.finfo(X.dtype).eps)).sum()
    Whx = Whx[:,:px] / sx[:px][_np.newaxis]
    # for Y
    Why, sy = _linalg.svd(Y.reshape(Y.shape[0],-1).real, full_matrices=False)[:2]
    # get rank
    py = (sy > (_np.max(sy) * _np.max([Y.shape[0],_np.prod(Y.shape[1:])]) *
          _np.finfo(Y.dtype).eps)).sum()
    Why = Why[:,:py] / sy[:py][_np.newaxis]
    # whiten the data
    X = _np.tensordot(Whx, X, axes = (0,0))
    Y = _np.tensordot(Why, Y, axes = (0,0))
    # get hilbert transform (if not complex)
    if not _np.iscomplexobj(X):
        X = _signal.hilbert(X, axis=1)
    if not _np.iscomplexobj(Y):
        Y = _signal.hilbert(Y, axis=1)
    # get the final number of filters
    num = _np.min([num, px, py])
    # determine if correlation coefficient is maximized or minimized
    if opt == 'max': sign = -1
    else: sign = 1
    # start optimization
    for i in range(num):
        if i == 0:
            # get first pair of filters
            # get best parameters and function values of each run
            optres = _np.array([
                     _minimize(func = _env_corr, fprime = None,
                               x0 = _np.random.random(px+py) * 2 -1,
                               args = (X, Y, sign, log, x_ind, y_ind),
                               m=100, approx_grad=False, iprint=1)[:2]
                     for k in range(bestof)])
            # somehow, _minimize sometimes returns the wrong function value,
            # so this is re-dertermined here
            for k in range(bestof):
                optres[k,1] = _env_corr(
                        optres[k,0],
                        X, Y, sign, log, x_ind, y_ind)[0]
            # determine the best
            best = _np.argmin(optres[:,1])
            # save results
            corr = [sign * optres[best,1]]
            filt = optres[best,0]
        else:
            # get consecutive pairs of filters
            # project data into null space of previous filters
            # this is done by getting the right eigenvectors of the filter
            # maxtrix corresponding to vanishing eigenvalues
            Bx = _linalg.svd(_np.atleast_2d(filt[:px].T),
                             full_matrices=True)[2][i:].T
            Xb = _np.tensordot(Bx,X, axes=(0,0))
            By = _linalg.svd(_np.atleast_2d(filt[px:].T),
                             full_matrices=True)[2][i:].T
            Yb = _np.tensordot(By,Y, axes=(0,0))
            # get best parameters and function values of each run
            optres = _np.array([
                     _minimize(func = _env_corr, fprime = None,
                               x0 = _np.random.random(px+py-2*i) * 2 -1,
                               args = (Xb, Yb, sign, log, x_ind, y_ind),
                               m=100, approx_grad=False, iprint=1)[:2]
                               for k in range(bestof)])
            # somehow, _minimize sometimes returns the wrong function value,
            # so this is re-dertermined here
            for k in range(bestof):
                optres[k,1] = _env_corr(
                        optres[k,0],
                        Xb, Yb, sign, log, x_ind, y_ind)[0]
            # determine the best
            best = _np.argmin(optres[:,1])
            # save results
            corr = corr + [sign * optres[best,1]]
            filt = _np.column_stack([filt,
                _np.hstack([Bx.dot(optres[best,0][:px-i]),
                            By.dot(optres[best,0][px-i:])])
                                    ])
    # project filters back into original (un-whitened) channel space
    Wx = Whx.dot(filt[:px])
    Wy = Why.dot(filt[px:])
    #normalize filters to have unit length
    Wx = Wx / _np.sqrt(_np.sum(Wx**2, 0))
    Wy = Wy / _np.sqrt(_np.sum(Wy**2, 0))
    return _np.array(corr), Wx, Wy

def _env_corr(wxy, Xa, Ya, sign=-1, log=True, x_ind=None, y_ind=None):
    """
    The cSPoC objective function: the correlation
    of amplitude envelopes

    Additionally, it returns the gradients of the objective functions
    with respect to each of the filter coefficients.

    Notes:
    ------
    The input datasets Xa and Ya are the analytic representations of the
    original datasets X and Y, hence they must be complex arrays.
    Xa and Ya can be either 2d numpy arrays of shape
    (channels x datapoints) or 3d array of shape
    (channels x datapoints x trials).
    For 3d arrays the average envelope in each trial is calculated if x_ind
    (or y_ind, respectively) is None. If they are set, the difference of
    the instantaneous amplitude envelope at x_ind/y_ind and the average
    envelope is calculated for each trial.
    If log == True, then the log transform is taken before the average
    inside the trial

    Input:
    ------
    -- wxy is an array of the concatenated filter coefficients for x and y
    -- Xa - numpy array - complex analytic representation of X
         Xa is the first Hilbert-transformed dataset of shape px x N (x tr),
         where px is the number of sensors, N the number of datapoints, tr
         the number of trials
    -- Ya is the second Hilbert-transformed dataset of shape py x N (x tr)
    -- sign {-1, 1} - the correlation coefficient is multiplied with this
                      number. If the result of this function is minimized
                      -1 should be used to find maximum correlation, 1
                      should be used to find maximal anti-correlation,
                      defaults to -1
    -- log {True, False} - compute the correlation between the log-
                           transformed envelopes, if datasets come in
                           epochs, then the log is taken before averaging
                           inside the epochs
    -- x_ind int - the time index (-Xa.shape[1] <= x_ind < Xa.shape[1]),
                   where the difference of the instantaneous envelope and
                   the average envelope is determined for Xa
    -- y_ind int - the time index (-Ya.shape[1] <= y_ind < Ya.shape[1]),
                   where the difference of the instantaneous envelope and
                   the average envelope is determined for Ya

    Output:
    -------
    -- c - float - the correlation coefficient of the amplitude envelopes
                   of X and Y multiplied by the value of \"sign\"
    -- c_der - numpy array - the gradient of c with respect to each of the
                             coefficients in wxy
    """
    assert isinstance(Xa, _np.ndarray), "Xa must be numpy array"
    assert _np.iscomplexobj(Xa), "Xa must be a complex-type numpy array" +\
                                 ", i.e. the analytic representaion of X"
    assert (Xa.ndim ==2 or Xa.ndim==3), "Xa must be 2D or 3D numpy array"
    assert isinstance(Ya, _np.ndarray), "Ya must be numpy array"
    assert _np.iscomplexobj(Ya), "Ya must be a complex-type numpy array" +\
                                 ", i.e. the analytic representation of Y"
    assert (Ya.ndim ==2 or Ya.ndim==3), "Ya must be 2D or 3D numpy array"
    assert Xa.shape[-1] == Ya.shape[-1], "Size of last dimension in Xa " +\
                                         "Ya must agree"
    p1 = Xa.shape[0]
    p2 = Ya.shape[0]
    assert len(wxy) == p1 + p2, "Length of wxy must equal the summed " + \
                                "number of variables in Xa and Ya"
    assert isinstance(log, bool), "\"log\" must be a boolean (True or False)"
    assert sign in [-1, 1], "\"sign\" must be -1 or 1"
    if x_ind != None:
        assert Xa.ndim == 3, "If x_ind is set, Xa must be 3d array!"
        assert isinstance(x_ind, int), "x_ind must be integer!"
        assert ((x_ind >= -Xa.shape[1]) and
                (x_ind < Xa.shape[1])), "x_ind must match the range of " +\
                                        "Xa.shape[1]"
    if y_ind != None:
        assert Ya.ndim == 3, "If y_ind is set, Ya must be 3d array!"
        assert isinstance(y_ind, int), "y_ind must be integer!"
        assert ((y_ind >= -Ya.shape[1]) and
                (y_ind < Ya.shape[1])), "y_ind must match the range of " +\
                                        "Ya.shape[1]"
    wx = wxy[:p1]
    wy = wxy[p1:]
    # filter signal spatially
    Xa_filt = _np.tensordot(wx, Xa, axes=(0,0))
    Ya_filt = _np.tensordot(wy, Ya, axes=(0,0))
    # get envelope of filtered signal
    x_env = _np.abs(Xa_filt)
    y_env = _np.abs(Ya_filt)
    # get derivatives of envelopes
    envx_derwx = ((Xa_filt.real * Xa.real +
                   Xa_filt.imag * Xa.imag) / x_env)
    envy_derwy = ((Ya_filt.real * Ya.real +
                   Ya_filt.imag * Ya.imag) / y_env)
    if log:
        envx_derwx = envx_derwx / x_env
        envy_derwy = envy_derwy / y_env
        x_env = _np.log(x_env)
        y_env = _np.log(y_env)
    if ((Xa.ndim == 3) and (x_ind != None)):
        envx_derwx = envx_derwx[:,x_ind] - envx_derwx.mean(1)
        x_env = x_env[x_ind] - x_env.mean(0)
    elif Xa.ndim == 3:
        envx_derwx = envx_derwx.mean(1)
        x_env = x_env.mean(0)
    if ((Ya.ndim == 3) and (y_ind != None)):
        envy_derwy = envy_derwy[:,y_ind] - envy_derwy.mean(1)
        y_env = y_env[y_ind] - y_env.mean(0)
    elif Ya.ndim == 3:
        envy_derwy = envy_derwy.mean(1)
        y_env = y_env.mean(0)
    # remove mean of envelopes and derivatives
    x_env = x_env - x_env.mean()
    y_env = y_env - y_env.mean()
    envx_derwx = envx_derwx - envx_derwx.mean(1)[:,_np.newaxis]
    envy_derwy = envy_derwy - envy_derwy.mean(1)[:,_np.newaxis]
    # get correlation of envelopes
    corr = ((x_env * y_env).mean() /
            _np.sqrt(_np.mean(x_env**2) * _np.mean(y_env**2)))
    # intermediate step to get derivatives
    delta_yx = y_env - x_env * _np.mean(x_env*y_env) / _np.mean(x_env**2)
    delta_xy = x_env - y_env * _np.mean(x_env*y_env) / _np.mean(y_env**2)
    #get complete derivatives
    der_wx = ((delta_yx * envx_derwx).mean(1) /
              _np.sqrt(_np.mean(x_env**2) * _np.mean(y_env**2)))
    der_wy = ((delta_xy * envy_derwy).mean(1) /
              _np.sqrt(_np.mean(x_env**2) * _np.mean(y_env**2)))
    return sign * corr, sign * _np.hstack([der_wx, der_wy])

def cSPoAC(X, tau=1, opt='max', num=1, log=True, bestof=15, x_ind=None):
    """
    canonical Soure Power Auto-Correlation analysis (cSPoAC)

    For the dataset X, find a linear filters wx, such
    that the correlation of the amplitude envelopes wx.T.dot(X[...,:-tau])
    and wx.T.dot(X[...,tau:]) is maximized, i.e. it seeks a spatial filter
    to maximize the auto-correlation of amplitude envelopes for a shift
    of tau (example for X being 2D).
    Alternatively tau can be an array of indices to X, such that
    X[...,tau[0]] and X[...,tau[1]] defince the lag.

    The solution is inspired by and derived by the original cSPoC-Analysis

    Reference:
    ----------
    Dahne, S., et al., Finding brain oscillations with power dependencies
    in neuroimaging data, NeuroImage (2014),
    http://dx.doi.org/10.1016/j.neuroimage.2014.03.075

    Notes:
    ------
    Dataset X can be either a 2d numpy array of shape
    (channels x datapoints) or 3d a array of shape
    (channels x datapoints x trials). For 2d array tau denotes a lag in
    the time domain, for 3d tau denotes a trial-wise lag.
    For 3d arrays the average envelope in each trial is calculated if x_ind
    is None. If it is set, the difference of the instantaneous amplitude
    at x_ind and the average envelope is calculated for each trial.

    If log == True, then the log transform is taken before the average
    inside the trial
    If X is of complex type, it is assumed that these is the analytic
    representations of X, i.e., the hilbert transform was applied before.

    The filters are in the columns of the filter matrices Wx and Wy,
    for 2d input the data can be filtered as:

    np.dot(Wx.T, X)

    for 3d input:

    np.tensordot(Wx, X, axes=(0,0))

    Input:
    ------
    -- X numpy array - the dataset of shape px x N (x tr), where px is the
                       number of sensors, N the number of data-points, tr
                       the number of trials. If X is of complex
                       type it is assumed that this already is the
                       analytic representation of X, i.e. the hilbert
                       transform already was applied.
    -- tau int or array of ints - the lag to calculate the autocorrelation,
                                  if X.ndim==2, this is a time-wise lag, if
                                  X.ndim==3, this is a trial-wise lag.
                                  Alternatively tau can be an array of ints,
                                  such that X[...,tau[0]] and X[...,tau[1]]
                                  are correlated.
    -- opt {'max', 'min', 'zero'} - determines whether the correlation coefficient
                            should be maximized - seeks for positive
                            correlations ('max', default);
                            or minimized - seeks for anti-correlations
                            ('min'), ('zero') seeks for zero correlation
    -- num int > 0 - determine the number of filters that will be derived.
                     This depends also on the rank of X, if X is 2d, the
                     number of filter pairs will be: min([num, rank(X)]).
                     If X is 3d, the array is flattened into a 2d array
                     before calculating the rank
    -- log {True, False} - compute the correlation between the log-
                           transformed envelopes, if datasets come in
                           epochs, then the log is taken before averaging
                           inside the epochs, defaults to True
    -- bestof int > 0 - the number of restarts for the optimization of the
                        individual filter pairs. The best filter over all
                        these restarts with random initializations is
                        chosen, defaults to 15.
    -- x_ind int - the time index (-X.shape[1] <= x_ind < X.shape[1]),
                   where the difference of the instantaneous envelope and
                   the average envelope is determined for X

    Output:
    -------
    corr - numpy array - the canonical correlations of the amplitude
                         envelopes for each filter
    Wx - numpy array - the filters for X, each filter is in a column of Wx
                       (if num==1: Wx is 1d)
    """
    #check input
    assert isinstance(X, _np.ndarray), "X must be numpy array"
    assert (X.ndim ==2 or X.ndim==3), "X must be 2D or 3D numpy array"
    if isinstance(tau, _np.ndarray):
        try:
            X[...,tau[0]]
            X[...,tau[1]]
        except:
            raise ValueError("""
                    If tau is an array, tau[0] and tau[1] must be subarrays
                    of valid indices to X defining a certain lag, i.e.,
                    the correlation between X[...,tau[0]] and X[...,tau[1]]
                     is optimized""")
    else:
        assert isinstance(tau, int), "tau must be integer-valued"
        assert ((tau > 0) and (tau < (X.shape[-1]-1))
            ), "tau must be >0 and smaller than the last dim of X " +\
               "minus 1."
        tau = _np.array([
            _np.arange(0,X.shape[-1]-tau,1),
            _np.arange(tau, X.shape[-1],1)
            ])
    assert opt in ['max', 'min', 'zero'], "\"opt\" must be \"max\", " +\
            "\"min\" or \"zero\""
    assert isinstance(num, int), "\"num\" must be integer > 0"
    assert num > 0, "\"num\" must be integer > 0"
    assert log in [True, False, 0, 1], "\"log\" must be a boolean (True " \
                                     + "or False)"
    assert isinstance(bestof, int), "\"bestof\" must be integer > 0"
    assert bestof > 0, "\"bestof\" must be integer > 0"
    if x_ind != None:
        assert X.ndim == 3, "If x_ind is set, X must be 3d array!"
        assert isinstance(x_ind, int), "x_ind must be integer!"
        assert ((x_ind >= -X.shape[1]) and
                (x_ind < X.shape[1])), "x_ind must match the range of " +\
                                       "X.shape[1]"
    # get whitening transformation for X
    Whx, sx = _linalg.svd(X.reshape(X.shape[0],-1).real, full_matrices=False)[:2]
    #get rank
    px = (sx > (_np.max(sx) * _np.max([X.shape[0],_np.prod(X.shape[1:])]) *
          _np.finfo(X.dtype).eps)).sum()
    Whx = Whx[:,:px] / sx[:px][_np.newaxis]
    # whiten the data
    X = _np.tensordot(Whx, X, axes = (0,0))
    # get hilbert transform
    if not _np.iscomplexobj(X):
        X = _signal.hilbert(X, axis=1)
    # get the final number of filters
    num = _np.min([num, px])
    # determine if correlation coefficient is maximized or minimized
    if opt == 'max': sign = -1
    elif opt == 'min': sign = 1
    elif opt == 'zero': sign = 0
    else: raise ValueError("\"opt\" must be \"max\", " +\
            "\"min\" or \"zero\"")
    # start optimization
    for i in range(num):
        if i == 0:
            # get first filter
            # get best parameters and function values of each run
            optres = _np.array([
                     _minimize(func = _env_corr_same, fprime = None,
                               x0 = _np.random.random(px) * 2 -1,
                               args = (X[...,tau[0]], X[...,tau[1]], sign, log,
                               x_ind, x_ind),
                               m=100, approx_grad=False, iprint=1)[:2]
                     for k in range(bestof)])
            # somehow, _minimize sometimes returns the wrong function value,
            # so this is re-dertermined here
            for k in range(bestof):
                optres[k,1] = _env_corr_same(
                        optres[k,0],
                        X[...,tau[0]], X[...,tau[1]], sign, log,
                               x_ind, x_ind)[0]
            # determine the best_result
            best = _np.argmin(optres[:,1])
            # save results
            if sign != 0:
                corr = [sign * optres[best,1]]
            else:
                corr = [optres[best,1]]
            filt = optres[best,0]
        else:
            # get consecutive pairs of filters
            # project data into null space of previous filters
            # this is done by getting the right eigenvectors of the filter
            # maxtrix corresponding to vanishing eigenvalues
            Bx = _linalg.svd(_np.atleast_2d(filt.T),
                             full_matrices=True)[2][i:].T
            Xb = _np.tensordot(Bx,X, axes=(0,0))
            # get best parameters and function values of each run
            optres = _np.array([
                     _minimize(func = _env_corr_same, fprime = None,
                               x0 = _np.random.random(px-i) * 2 -1,
                               args = (Xb[...,tau[0]], Xb[...,tau[1]], sign, log,
                               x_ind, x_ind),
                               m=100, approx_grad=False, iprint=1)[:2]
                               for k in range(bestof)])
            # somehow, _minimize sometimes returns the wrong function value,
            # so this is re-dertermined here
            for k in range(bestof):
                optres[k,1] = _env_corr_same(
                        optres[k,0],
                        Xb[...,tau[0]], Xb[...,tau[1]], sign, log,
                               x_ind, x_ind)[0]
            # determine the best result
            best = _np.argmin(optres[:,1])
            # save results
            if sign != 0:
                corr = corr + [sign * optres[best,1]]
            else:
                corr = corr + [optres[best,1]]
            filt = _np.column_stack([filt, Bx.dot(optres[best,0])])
    # project filters back into original (un-whitened) channel space
    Wx = Whx.dot(filt)
    #normalize filters to have unit length
    Wx = Wx / _np.sqrt(_np.sum(Wx**2, 0))
    return _np.array(corr), Wx

def _env_corr_same(wxy, Xa, Ya, sign=-1, log=True, x_ind=None, y_ind=None):
    """
    The cSPoC objective function with same filters for both data sets:
    the correlation of amplitude envelopes

    Additionally, it returns the gradients of the objective function
    with respect to each of the filter coefficients.

    Notes:
    ------
    The input datasets Xa and Ya are the analytic representations of the
    original datasets X and Y, hence they must be complex arrays.
    Xa and Ya can be either 2d numpy arrays of shape
    (channels x datapoints) or 3d array of shape
    (channels x datapoints x trials).
    For 3d arrays the average envelope in each trial is calculated if x_ind
    (or y_ind, respectively) is None. If they are set, the difference of
    the instantaneous amplitude envelope at x_ind/y_ind and the average
    envelope is calculated for each trial.
    If log == True, then the log transform is taken before the average
    inside the trial

    Input:
    ------
    -- wxy is the array of shared filter coefficients for x and y
    -- Xa - numpy array - complex analytic representation of X
         Xa is the first Hilbert-transformed dataset of shape px x N (x tr),
         where px is the number of sensors, N the number of datapoints, tr
         the number of trials
    -- Ya is the second Hilbert-transformed dataset of shape py x N (x tr)
    -- sign {-1, 1} - the correlation coefficient is multiplied with this
                      number. If the result of this function is minimized
                      -1 should be used to find maximum correlation, 1
                      should be used to find maximal anti-correlation,
                      defaults to -1
    -- log {True, False} - compute the correlation between the log-
                           transformed envelopes, if datasets come in
                           epochs, then the log is taken before averaging
                           inside the epochs
    -- x_ind int - the time index (-Xa.shape[1] <= x_ind < Xa.shape[1]),
                   where the difference of the instantaneous envelope and
                   the average envelope is determined for Xa
    -- y_ind int - the time index (-Ya.shape[1] <= y_ind < Ya.shape[1]),
                   where the difference of the instantaneous envelope and
                   the average envelope is determined for Ya

    Output:
    -------
    -- c - float - the correlation coefficient of the amplitude envelopes
                   of X and Y multiplied by the value of \"sign\"
    -- c_der - numpy array - the gradient of c with respect to each of the
                             coefficients in wxy
    """
    assert isinstance(Xa, _np.ndarray), "Xa must be numpy array"
    assert _np.iscomplexobj(Xa), "Xa must be a complex-type numpy array" +\
                                 ", i.e. the analytic representaion of X"
    assert (Xa.ndim ==2 or Xa.ndim==3), "Xa must be 2D or 3D numpy array"
    assert isinstance(Ya, _np.ndarray), "Ya must be numpy array"
    assert _np.iscomplexobj(Ya), "Ya must be a complex-type numpy array" +\
                                 ", i.e. the analytic representation of Y"
    assert (Ya.ndim ==2 or Ya.ndim==3), "Ya must be 2D or 3D numpy array"
    assert Xa.shape[-1] == Ya.shape[-1], "Size of last dimension in Xa " +\
                                         "Ya must agree"
    p1 = Xa.shape[0]
    p2 = Ya.shape[0]
    assert p1 == p2, 'Dimensionality of Xa and Ya must agree for cSPoc' +\
                     ' with same filters'
    assert len(wxy) == p1, "Length of wxy must equal the" + \
                           " number of variables in Xa and Ya"
    assert isinstance(log, bool), "\"log\" must be a boolean (True or False)"
    assert sign in [-1, 1, 0], "\"sign\" must be -1, 1, or 0"
    if x_ind != None:
        assert Xa.ndim == 3, "If x_ind is set, Xa must be 3d array!"
        assert isinstance(x_ind, int), "x_ind must be integer!"
        assert ((x_ind >= -Xa.shape[1]) and
                (x_ind < Xa.shape[1])), "x_ind must match the range of " +\
                                        "Xa.shape[1]"
    if y_ind != None:
        assert Ya.ndim == 3, "If y_ind is set, Ya must be 3d array!"
        assert isinstance(y_ind, int), "y_ind must be integer!"
        assert ((y_ind >= -Ya.shape[1]) and
                (y_ind < Ya.shape[1])), "y_ind must match the range of " +\
                                        "Ya.shape[1]"
    # filter signal spatially
    Xa_filt = _np.tensordot(wxy, Xa, axes=(0,0))
    Ya_filt = _np.tensordot(wxy, Ya, axes=(0,0))
    # get envelope of filtered signal
    x_env = _np.abs(Xa_filt)
    y_env = _np.abs(Ya_filt)
    # get derivatives of envelopes
    envx_derwx = ((Xa_filt.real * Xa.real +
                   Xa_filt.imag * Xa.imag) / x_env)
    envy_derwy = ((Ya_filt.real * Ya.real +
                   Ya_filt.imag * Ya.imag) / y_env)
    if log:
        envx_derwx = envx_derwx / x_env
        envy_derwy = envy_derwy / y_env
        x_env = _np.log(x_env)
        y_env = _np.log(y_env)
    if ((Xa.ndim == 3) and (x_ind != None)):
        envx_derwx = envx_derwx[:,x_ind] - envx_derwx.mean(1)
        x_env = x_env[x_ind] - x_env.mean(0)
    elif Xa.ndim == 3:
        envx_derwx = envx_derwx.mean(1)
        x_env = x_env.mean(0)
    if ((Ya.ndim == 3) and (y_ind != None)):
        envy_derwy = envy_derwy[:,y_ind] - envy_derwy.mean(1)
        y_env = y_env[y_ind] - y_env.mean(0)
    elif Ya.ndim == 3:
        envy_derwy = envy_derwy.mean(1)
        y_env = y_env.mean(0)
    # remove mean of envelopes and derivatives
    x_env = x_env - x_env.mean()
    y_env = y_env - y_env.mean()
    envx_derwx = envx_derwx - envx_derwx.mean(1)[:,_np.newaxis]
    envy_derwy = envy_derwy - envy_derwy.mean(1)[:,_np.newaxis]
    # numerator of correlation
    num = _np.mean(x_env * y_env)
    # derivative of numerator
    num_d = _np.mean(envx_derwx*y_env + x_env*envy_derwy,1)
    # denominator of correlation
    denom = _np.sqrt(_np.mean(x_env**2) * _np.mean(y_env**2))
    # derivative of denominator
    denom_d = (
     (_np.mean(x_env*envx_derwx,1)*_np.mean(y_env**2) +
      _np.mean(x_env**2)*_np.mean(y_env*envy_derwy,1)
     ) /
     _np.sqrt(_np.mean(x_env**2) * _np.mean(y_env**2)))
    #final correlation
    corr = num / denom
    #final derivative
    corr_d = (num_d*denom - num*denom_d) / denom**2
    if sign == 0:
        return _np.sign(corr)*corr, _np.sign(corr)*corr_d
    else:
        return sign*corr, sign*corr_d

def cSPoAvgC(X, opt='max', num=1, log=True, bestof=15):
    """
    canonical Soure Power Average Correlation analysis (cSPoAvgC)

    For the dataset X, find a linear filters wx, such
    that the average correlation of the amplitude envelopes of wx.T.dot(X)
    and (wx.T.dot(X)).mean(-1) is maximized, i.e. it seeks a spatial filter
    to maximize the correlation of amplitude envelopes and their average.

    The solution is inspired by and derived by the original cSPoC-Analysis

    Reference:
    ----------
    Dahne, S., et al., Finding brain oscillations with power dependencies
    in neuroimaging data, NeuroImage (2014),
    http://dx.doi.org/10.1016/j.neuroimage.2014.03.075

    Notes:
    ------
    Dataset X must be a 3d a array of shape
    (channels x datapoints x trials).

    If log == True, then the log transform is taken before the average
    inside the trial

    If X is of complex type, it is assumed that these is the analytic
    representations of X, i.e., the hilbert transform was applied before.

    The filters are in the columns of the filter matrices Wx

    The input data can be filtered as:

    np.tensordot(Wx, X, axes=(0,0))

    Input:
    ------
    -- X numpy array - the dataset of shape px x N x tr, where px is the
                       number of sensors, N the number of data-points, tr
                       the number of trials. If X is of complex
                       type it is assumed that this already is the
                       analytic representation of X, i.e. the hilbert
                       transform already was applied.
    -- opt {'max', 'min', 'zero'} - determines whether the correlation coefficient
                            should be maximized - seeks for positive
                            correlations ('max', default);
                            or minimized - seeks for anti-correlations
                            ('min'), ('zero') seeks for zero correlation
    -- num int > 0 - determine the number of filters that will be derived.
                     This depends also on the rank of X, if X is 2d, the
                     number of filter pairs will be: min([num, rank(X)]).
                     If X is 3d, the array is flattened into a 2d array
                     before calculating the rank
    -- log {True, False} - compute the correlation between the log-
                           transformed envelopes, if datasets come in
                           epochs, then the log is taken before averaging
                           inside the epochs, defaults to True
    -- bestof int > 0 - the number of restarts for the optimization of the
                        individual filter pairs. The best filter over all
                        these restarts with random initializations is
                        chosen, defaults to 15.
    Output:
    -------
    corr - numpy array - the canonical correlations of the amplitude
                         envelopes for each filter
    Wx - numpy array - the filters for X, each filter is in a column of Wx
                       (if num==1: Wx is 1d)
    """
    #check input
    assert isinstance(X, _np.ndarray), "X must be numpy array"
    assert (X.ndim==3), "X must be 3D numpy array"
    assert opt in ['max', 'min', 'zero'], "\"opt\" must be \"max\", " +\
            "\"min\" or \"zero\""
    assert isinstance(num, int), "\"num\" must be integer > 0"
    assert num > 0, "\"num\" must be integer > 0"
    assert log in [True, False, 0, 1], "\"log\" must be a boolean (True " \
                                     + "or False)"
    assert isinstance(bestof, int), "\"bestof\" must be integer > 0"
    assert bestof > 0, "\"bestof\" must be integer > 0"
    # get whitening transformation for X
    Whx, sx = _linalg.svd(X.reshape(X.shape[0],-1).real, full_matrices=False)[:2]
    #get rank
    px = (sx > (_np.max(sx) * _np.max([X.shape[0],_np.prod(X.shape[1:])]) *
          _np.finfo(X.dtype).eps)).sum()
    Whx = Whx[:,:px] / sx[:px][_np.newaxis]
    # whiten the data
    X = _np.tensordot(Whx, X, axes = (0,0))
    # get hilbert transform
    if not _np.iscomplexobj(X):
        X = _signal.hilbert(X, axis=1)
    # get the final number of filters
    num = _np.min([num, px])
    # determine if correlation coefficient is maximized or minimized
    if opt == 'max': sign = -1
    elif opt == 'min': sign = 1
    elif opt == 'zero': sign = 0
    else: raise ValueError("\"opt\" must be \"max\", " +\
            "\"min\" or \"zero\"")
    # start optimization
    for i in range(num):
        if i == 0:
            # get first filter
            # get best parameters and function values of each run
            optres = _np.array([
                     _minimize(func = _env_corr_avg, fprime = None,
                               x0 = _np.random.random(px) * 2 -1,
                               args = (X, sign, log), m=100,
                                   approx_grad=False, iprint=1)[:2]
                     for k in range(bestof)])
            # somehow, _minimize sometimes returns the wrong function value,
            # so this is re-dertermined here
            for k in range(bestof):
                optres[k,1] = _env_corr_avg(
                        optres[k,0],
                        X, sign, log)[0]
            # determine the best_result
            best = _np.argmin(optres[:,1])
            # save results
            if sign != 0:
                corr = [sign * optres[best,1]]
            else:
                corr = [optres[best,1]]
            filt = optres[best,0]
        else:
            # get consecutive pairs of filters
            # project data into null space of previous filters
            # this is done by getting the right eigenvectors of the filter
            # maxtrix corresponding to vanishing eigenvalues
            Bx = _linalg.svd(_np.atleast_2d(filt.T),
                             full_matrices=True)[2][i:].T
            Xb = _np.tensordot(Bx,X, axes=(0,0))
            # get best parameters and function values of each run
            optres = _np.array([
                     _minimize(func = _env_corr_avg, fprime = None,
                               x0 = _np.random.random(px-i) * 2 -1,
                               args = (Xb, sign, log),
                               m=100, approx_grad=False, iprint=1)[:2]
                               for k in range(bestof)])
            # somehow, _minimize sometimes returns the wrong function value,
            # so this is re-dertermined here
            for k in range(bestof):
                optres[k,1] = _env_corr_avg(
                        optres[k,0],
                        Xb, sign, log)[0]
            # determine the best result
            best = _np.argmin(optres[:,1])
            # save results
            if sign != 0:
                corr = corr + [sign * optres[best,1]]
            else:
                corr = corr + [optres[best,1]]
            filt = _np.column_stack([filt, Bx.dot(optres[best,0])])
    # project filters back into original (un-whitened) channel space
    Wx = Whx.dot(filt)
    #normalize filters to have unit length
    Wx = Wx / _np.sqrt(_np.sum(Wx**2, 0))
    return _np.array(corr), Wx

def _env_corr_avg(wxy, Xa, sign=-1, log=True):
    """
    The cSPoC objective function to maximize correlation  of single-trial
    envelopes to average envelope.

    Additionally, it returns the gradients of the objective function
    with respect to each of the filter coefficient.

    Notes:
    ------
    The input datasets Xa is the analytic representations of the
    original datasets X, hence it must be a complex array.
    Xa must be a 3d array of shape (channels x datapoints x trials).
    If log == True, then the log transform is taken before the average
    inside the trial

    Input:
    ------
    -- wxy is the array of shared filter coefficients for x and y
    -- Xa - numpy array - complex analytic representation of X
         Xa is the first Hilbert-transformed dataset of shape px x N x tr,
         where px is the number of sensors, N the number of datapoints, tr
         the number of trials
    -- sign {-1, 1} - the correlation coefficient is multiplied with this
                      number. If the result of this function is minimized
                      -1 should be used to find maximum correlation, 1
                      should be used to find maximal anti-correlation,
                      defaults to -1
    -- log {True, False} - compute the correlation between the log-
                           transformed envelopes, if datasets come in
                           epochs, then the log is taken before averaging
                           inside the epochs
    Output:
    -------
    -- c - float - the correlation coefficient of the amplitude envelopes
                   of X and Y multiplied by the value of \"sign\"
    -- c_der - numpy array - the gradient of c with respect to each of the
                             coefficients in wxy
    """
    assert isinstance(Xa, _np.ndarray), "Xa must be numpy array"
    assert _np.iscomplexobj(Xa), "Xa must be a complex-type numpy array" +\
                                 ", i.e. the analytic representaion of X"
    assert (Xa.ndim==3), "Xa must be  3D numpy array"
    p1 = Xa.shape[0]
    assert len(wxy) == p1, "Length of wxy must equal the" + \
                           " number of variables in Xa"
    assert isinstance(log, bool), "\"log\" must be a boolean (True or False)"
    assert sign in [-1, 1, 0], "\"sign\" must be -1, 1, or 0"
    # filter signal spatially
    Xa_filt = _np.tensordot(wxy, Xa, axes=(0,0))
    # get envelope of filtered signal
    x_env = _np.abs(Xa_filt)
    # get derivatives of envelopes
    envx_derwx = ((Xa_filt.real * Xa.real +
                   Xa_filt.imag * Xa.imag) / x_env)
    if log:
        envx_derwx = envx_derwx / x_env
        x_env = _np.log(x_env)
    # remove mean of envelopes and derivatives
    x_env = x_env - x_env.mean(0)
    envx_derwx = envx_derwx - envx_derwx.mean(1)[:,_np.newaxis]
    # numerator of correlation
    num = _np.mean(
                (x_env * x_env.mean(-1)[:,_np.newaxis]),
                0)
    # derivative of numerator
    num_d = _np.mean(
                envx_derwx*x_env.mean(-1)[:,_np.newaxis] +
                x_env*envx_derwx.mean(-1)[:,:,_np.newaxis],
                1)
    # denominator of correlation
    denom = _np.sqrt(_np.mean(x_env**2,0) *
            _np.mean(x_env.mean(-1)**2,0))
    # derivative of denominator
    denom_d = (
     (_np.mean(x_env*envx_derwx,1)*_np.mean(x_env.mean(-1)**2,0)) +
     _np.mean(x_env**2,0)*_np.mean(x_env.mean(-1)*
         envx_derwx.mean(-1),1)[:,_np.newaxis]
     ) / denom
    #final correlation
    corr = _np.mean(num / denom)
    #final derivative
    corr_d = ((num_d*denom - num*denom_d) / denom**2).mean(-1)
    if sign == 0:
        return _np.sign(corr)*corr, _np.sign(corr)*corr_d
    else:
        return sign*corr, sign*corr_d

