"""
This implements Phase Coupling Optimization (PCO)

A citable reference is no ready yet, but (hopefully) soon.

Hiddem submodule of the Modular EEg Toolkit - MEET for Python.
(Imported by spatfilt module).
"""
from __future__ import division
from scipy.optimize import fmin_l_bfgs_b as _minimize

if __name__ != '__main__':
    from . import _np
    from . import _linalg
else:
    import numpy as _np
    import scipy.linalg as _linalg

def PCOa(a, Y, num=1, bestof=15):
    '''
    Phase Amplitude Coupling Optimization, variant with provided amplitude

    It maximizes the length of the "mean vector" and
    returns the filter coefficients w

    Input:
    ------
    a - (1d numpy array, floats > 0) amplitudes
    Y - (2d numpy array, complex) analytic representation of signal,
        channels x datapoints
    num - (int > 0) - determines the number of filters that will be
                      derived. This depends also on the rank of Y, the final
                      number of filters will be min([num, rank(Y)]),
                      defaults to 1
    bestof (int > 0) - the number of restarts for the optimization of the
                       individual filters. The best filter over all
                       these restarts with random initializations is
                       chosen, defaults to 15.

    Output:
    -------
    vlen - numpy array - the length of the mean vector for each filter
    Wy - numpy array - the filters for Y, each filter is in a column of Wy
                       (if num==1: Wx is 1d)
    '''
    ############################
    # test all input arguments #
    ############################
    try:
        a = _np.asarray(a).astype(float)
    except:
        raise TypeError('a must be iterable of floats')
    if not a.ndim == 1:
        raise ValueError('a must be 1-dimensional')
    ###
    try:
        Y = _np.asarray(Y)
    except:
        raise TypeError('Y must be iterable')
    if not _np.iscomplexobj(Y):
        raise TypeError('Y must be complex valued')
    if not Y.ndim == 2:
        raise ValueError('Y must be 2d')
    if not Y.shape[1] == len(a):
        raise ValueError('Number of points in Y must match the length of a')
    ###
    if not isinstance(num, int):
        raise TypeError('num must be integer')
    if not num > 0:
        raise ValueError('num must be > 0')
    ###
    if not isinstance(bestof, int):
        raise TypeError('bestof must be integer')
    if not bestof > 0:
        raise ValueError('bestof must be > 0')
    #####################################################
    # normalize a to have zero mean and a variance of 1 #
    #####################################################
    a = (a - a.mean())/a.std()
    ######################################
    # Whiten the (real part of the) data #
    ######################################
    Why, sy = _linalg.svd(Y.real, full_matrices=False)[:2]
    # get rank
    py = (sy > (_np.max(sy) * _np.max([Y.shape[0],_np.prod(Y.shape[1:])]) *
          _np.finfo(Y.dtype).eps)).sum()
    Why = Why[:,:py] / sy[:py][_np.newaxis]
    # whiten for the real part of the data
    Y = _np.dot(Why.T, Y)
    # get the final number of filters
    num = _np.min([num, py])
    ######################
    # start optimization #
    ######################
    for i in xrange(num):
        if i == 0:
            # get first filter
            # get best parameters and function values of each run
            optres = _np.array([
                     _minimize(func = _PCOa_obj_der, fprime = None,
                               x0 = _np.random.random(py) * 2 -1,
                               args = (a,Y,-1,True,False),
                               m=100, approx_grad=False, iprint=0)[:2]
                     for k in xrange(bestof)])
            # somehow, _minimize sometimes returns the wrong function value,
            # so this is re-dertermined here
            for k in xrange(bestof):
                optres[k,1] = _PCOa_obj_der(
                        optres[k,0],
                        a,Y,-1,False,False)
            # determine the best
            best = _np.argmin(optres[:,1])
            # save results
            vlen = [optres[best,1]]
            filt = optres[best,0]
        else:
            # get consecutive pairs of filters
            # project data into null space of previous filters
            # this is done by getting the right eigenvectors of the filter
            # maxtrix corresponding to vanishing eigenvalues
            By = _linalg.svd(_np.atleast_2d(filt.T),
                             full_matrices=True)[2][i:].T
            Yb = _np.dot(By.T,Y)
            # get best parameters and function values of each run
            optres = _np.array([
                     _minimize(func = _PCOa_obj_der, fprime = None,
                               x0 = _np.random.random(py - i) * 2 -1,
                               args = (a,Yb,-1,True,False),
                               m=100, approx_grad=False, iprint=0)[:2]
                     for k in xrange(bestof)])
            # somehow, _minimize sometimes returns the wrong function value,
            # so this is re-determined here
            for k in xrange(bestof):
                optres[k,1] = _PCOa_obj_der(
                        optres[k,0],
                        a,Yb,-1,False,False)
            # determine the best
            best = _np.argmin(optres[:,1])
            # save results
            vlen = vlen + [optres[best,1]]
            filt = _np.column_stack([
                filt,
                By.dot(optres[best,0])])
    # project filters back into original (un-whitened) channel space
    Wy = Why.dot(filt)
    #normalize filters to have unit length
    Wy = Wy / _np.sqrt(_np.sum(Wy**2, 0))
    return -1*_np.array(vlen), Wy

def _PCOa_obj_der(w, a, Y, sign=-1, return_der=True, check_input=True):
    '''
    Internal function! Do not use directly!

    It returns the length of the "mean vector" and
    if return_der is True also the partial derivatives with respect to
    each filter coefficient

    Input:
    ------
    w - (1d numpy array, float) filter coefficients
    a - (1d numpy array, floats > 0) amplitudes, it is assumed that
        a has a mean of 0 and variance of 1
    Y - (2d numpy array, complex) analytic representation of signal,
        channels x datapoints
    sign {-1, 1} - the result is multiplied with this number. If the
        result of this function is minimized -1 should be used to find
        maximum vector length.
    return_der - (boolean) if the partial derivatives should be returned
        besides vlen
    check_input - (boolean) if the input should be ckecked - function is
        faster without, but might fail without helpful error messages

    Output:
    -------
    vlen -(float) length of mean vector
    vlen_der - (array of floats) - partial derivatives of v_len with
               respect to each filter coefficient (only if
               return_der == True)
    '''
    ############################
    # test all input arguments #
    ############################
    if not isinstance(check_input, bool):
        raise TypeError('check_input must be True or False')
    if check_input:
        try:
            w = _np.asarray(w).astype(float)
        except:
            raise TypeError('w must be iterable of floats')
        if not w.ndim == 1:
            raise ValueError('w must be 1-dimensional')
        ###
        try:
            a = _np.asarray(a).astype(float)
        except:
            raise TypeError('a must be iterable of floats')
        if not a.ndim == 1:
            raise ValueError('a must be 1-dimensional')
        if not _np.isclose(a.mean(),0):
            raise ValueError('the mean of a must be 0')
        if not _np.isclose(a.std(),1):
            raise ValueError('the standard deviation of a must be 1')
        ###
        try:
            Y = _np.asarray(Y)
        except:
            raise TypeError('Y must be iterable')
        if not _np.iscomplexobj(Y):
            raise TypeError('Y must be complex valued')
        if not Y.ndim == 2:
            raise ValueError('Y must be 2d')
        if not Y.shape[0] == len(w):
            raise ValueError('the number of channels in Y must agree ' +
            'with the number of filter coefficients in w')
        if not Y.shape[1] == len(a):
            raise ValueError('the number of points in Y must agree ' +
            'with the number of points in a')
        ###
        if not sign in [-1,1]:
            raise ValueError('sign must be -1 or 1')
        ###
        if not isinstance(return_der, bool):
            raise TypeError('return_der must be True or False')
    #######################################################################
    # start calculation of mean vector length and its partial derivatives #
    #######################################################################
    filt = w.dot(Y)
    filt_norm = filt/_np.abs(filt)
    ####################################
    # result of the objective function #
    ####################################
    sum1_no_square = _np.mean(a * filt_norm.real, -1)
    sum2_no_square = _np.mean(a * filt_norm.imag, -1)
    obj = _np.sqrt(sum1_no_square**2 + sum2_no_square**2)
    if return_der:
        # partial derivative of phase
        phase_dwi = ((-Y.real*filt.imag + Y.imag*filt.real)/
                (filt.real**2 + filt.imag**2))
        # derivative of first summand
        sum1_d = 2*sum1_no_square*_np.einsum('...j,ij->i',
                -a*filt_norm.imag, phase_dwi)/len(a)
        # derivative of second summand
        sum2_d = 2*sum2_no_square*_np.einsum('...j,ij->i',
                a*filt_norm.real, phase_dwi)/len(a)
        # derivative of sum
        sum_d = sum1_d + sum2_d
        # derivative including square root
        obj_d = sum_d/(2*obj)
        #return both objective function and partial derivatives
        # return negative if the function should be minimized
        return sign*obj, sign*obj_d
    else:
        return sign*obj

if __name__ == "__main__":
    N_ch = 20
    N_dp = 1000000
    N_amp =1000
    s_rate = 200
    # define the inputs
    a = _np.random.rand(N_amp)
    #a[-N_dp//4:] = 1000
    # normalize a
    a = (a - a.mean())/a.std()
    # get marker
    marker = _np.sort(_np.random.randint(low=0, high=N_dp, size=N_amp))
    # get random phases
    p = _np.random.uniform(-_np.pi, _np.pi, N_amp)
    #
    w = _np.random.rand(N_ch)
    raw_data = _np.random.rand(N_ch,N_dp)
    import meet
    import scipy.signal
    x = meet.iir.butterworth(raw_data, fp=[10,12], fs = [8,14],
            s_rate=s_rate)
    x = scipy.signal.hilbert(x, axis=-1)
    y = meet.iir.butterworth(raw_data, fp=[5,20], fs = [8.8,13.2],
            s_rate=s_rate)
    #############################################
    # test the output of the gradient functions #
    #############################################
    # get analytical results
    vlen_a, analytical_der_a = _PCOa_obj_der(w, a, x[:,marker])
    # determine gradient as finite difference and compare to analytical
    # gradients
    dx = 1E-10
    dw = _np.zeros([N_ch,N_ch],float)
    dw[range(N_ch), range(N_ch)] += dx
    vlen_a_fd_grad = _np.array([(
        _PCOa_obj_der(w+dw_now, a, x[:,marker],-1, False) -
        _PCOa_obj_der(w       , a, x[:,marker], -1, False)) /
        dx for dw_now in dw])
    #plot to compare results
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(analytical_der_a, 'bo-', label='analytic gradient',
            markersize=6)
    ax1.plot(vlen_a_fd_grad, 'r*-', label='finite-difference gradient',
            markersize=6)
    ax1.set_xlabel('filter coefficient')
    ax1.set_ylabel('partial derivative')
    ax1.grid()
    ax1.legend(loc='best')
    ax1.set_title('gradients of PCOa')
    fig.tight_layout()
    plt.show()
    ########################
    # run the optimization #
    ########################
    vlen, wy = PCOa(a, x[:,marker], num=4, bestof=50)
    # check if the vector lenght when applying the filter matches the
    # output of the function
    vlen2 = _np.array([_PCOa_obj_der(w_now, a, x[:,marker], 1, False)
        for w_now in wy.T])
    #plot to compare results
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(vlen, 'bo-', label='vlen, function output',
            markersize=6)
    ax.plot(vlen2, 'r*-', label='vlen2, determined separately',
            markersize=6)
    ax.set_xlabel('filter index')
    ax.set_ylabel('lenght of mean vector')
    ax.grid()
    ax.legend(loc='best')
    fig.tight_layout()
    plt.show()
