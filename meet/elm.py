'''
Extreme Learning Machine Classification

Submodule of the Modular EEg Toolkit - MEET for Python.

This module implements regularized Extreme Learning Machine
Classification and Weighted Extreme Learning Machine Classification.

Classification is implemented in the ClassELM class

For faster execution of dot product the module dot_new is imported since
it avoids the need of temporary copy and calls fblas directly.
The code is available here: http://pastebin.com/raw.php?i=M8TfbURi 
In future this will be available in numpy directly:
https://github.com/numpy/numpy/pull/2730

1. Extreme Learning Machine for Regression and Multiclass Classification
Guang-Bin Huang, Hongming Zhou, Xiaojian Diang, Rui Zhang
IEEE Transactions of Systems, Man and Cybernetics - Pat B: Cybernetics,
Vol. 42, No. 2. April 2012

2. Weighted extreme learning machine for imbalance learning.
Weiwei Zong, Guang-Bin Huang, Yiqiang Chen
Neurocomputing 101 (2013) 229-242

Author & Contact
----------------
Written by Gunnar Waterstraat
email: gunnar[dot]waterstraat[at]charite.de
'''
from __future__ import division

from . import _np
from . import _linalg

try: from _dot_new import dot as _dot
except: _dot = _np.dot

def accuracy(conf_matrix):
    '''
    Measure of the performance of the classifier.
    The Accuracy is the proportion of correctly classified items in
    relation to the total number of items.

    You should be aware that this is very sensitive to imbalanced data
    (data with very unequal sizes of each class):
    Imagine a sample with 99% of the items belonging to class 0 and 1%
    of items belonging to class 1. A classifier might have an accuracy
    of 99% by just assigning all items to class 0. However, the
    sensitivity for class 1 is 0% in that case. It depends on your needs
    if this is acceptable or not.
    
    Input:
    ------
    conf_matrix - shape ny x ny, where ny is the number of classes
                  the rows belong to the actual, the columns to the
                  predicted class: item ij is hence predicted as class
                  j, while it would have belonged to class i
    
    Output:
    -------
    float - the accuracy
    '''
    return _np.trace(conf_matrix) / float(conf_matrix.sum(None))

def G_mean(conf_matrix):
    '''
    The G-mean is the geometric mean of the per-class-sensitivities. It
    is much more stable to imbalance of the dataset than the global
    accuray. However it depends on your needs, which measure of
    performance of the classifier to use.
    
    Input:
    ------
    conf_matrix - shape ny x ny, where ny is the number of classes
                 the rows belong to the actual, the columns to the
                 predicted class: item ij is hence predicted as class j,
                 while it would have belonged to class i
    
    Output:
    -------
    the geometric mean of per-class sensitivities
    '''
    from scipy.stats.mstats import gmean as _gmean
    # get per-class accuracy - which is the number of items correctly
    # classified in this class
    # in relation to total number of items actually in this class
    per_class = _np.diag(conf_matrix) / conf_matrix.sum(1).astype(float)
    # get geometric average
    if _np.any(per_class == 0): return 0.
    else: return _gmean(per_class)

def Matthews(conf_matrix):
    '''
    The Matthews correlation coefficient is used in machine learning as
    a measure of the quality of binary (two-class) classifications. It
    takes into account true and false positives and negatives and is
    generally regarded as a balanced measure which can be used even if
    the classes are of very different sizes. The MCC is in essence a
    correlation coefficient between the observed and predicted binary
    classifications; it returns a value between -1 and +1. A coefficient
    of +1 represents a perfect prediction, 0 no better than random
    prediction and -1 indicates total disagreement between prediction
    and observation.

    Source: Wikipedia (2013-09-25)
    
    Input:
    ------
    conf_matrix - shape 2 x 2, where 2 is the number of classes
                 the rows belong to the actual, the columns to the
                 predicted class: item ij is hence predicted as class j,
                 while it would have belonged to class i
                 AN ERROR IS THROWN IF THE SHAPE OF THE MATRIX IS NOT
                 CORRECT

    Output:
    -------
    float - the the Matthews Correlation Coefficient
    '''
    try: conf_matrix.shape
    except:
        raise TypeError('conf_matrix must be numpy array or' +
                        'numpy matrix')
    assert conf_matrix.shape == (2,2),'conf_matrix must be of shape 2x2'
    TN, FP, FN, TP = conf_matrix.ravel()
    try: MCC = _np.exp(_np.log(long(TP)*long(TN) - long(FP)*long(FN)) -
            (_np.log(TP+FP)+_np.log(TP+FN)+_np.log(TN+FP) + 
                _np.log(TN+FN)) / 2.)
    except: MCC = 0
    return MCC

def PPV(conf_matrix):
    '''
    Calculate the Positive Predictive Value

    Input:
    ------
    conf_matrix - shape 2 x 2, where 2 is the number of classes
                 the rows belong to the actual, the columns to the
                 predicted class: item ij is hence predicted as class j,
                 while it would have belonged to class i
                 AN ERROR IS THROWN IF THE SHAPE OF THE MATRIX IS NOT
                 CORRECT

    Output:
    -------
    float - the PPV
    '''
    try: conf_matrix.shape
    except:
        raise TypeError('conf_matrix must be numpy array or numpy' +
                        'matrix')
    if not conf_matrix.shape == (2,2):
        raise ValueError('conf_matrix must be of shape 2x2')
    TN, FP, FN, TP = conf_matrix.ravel()
    PPV = TP / float(TP+FP)
    return PPV

def ssk_cv(data, labels, folds=3):
    '''
    Cut data into folds with method:
    shuffled, stratified, k-folds cross-validation
    
    Input:
    ------
    data - numpy array - shape n x p, with n items anf p features
    
    Output:
    -------
    returns a list, with each list-element including the indices of one
    fold
    '''
    from itertools import chain
    N = data.shape[0] # number of samples and features
    # shuffle data and labels - however, their relation shouldn't be
    # changed
    rarray = _np.random.random(N).argsort()
    data = data[rarray]
    labels = labels[rarray]
    # sort according to the labels
    order = labels.argsort()
    data = data[order]
    labels = labels[order]
    rarray = rarray[order] # the original indices
    # find number of classes and count
    cc, ci = _np.unique(labels, return_index = True) # the classes and
    #first index of each class
    ci = _np.append(ci,N)
    #find the number of items of each class in each fold
    start_stop = _np.array([_np.round(_np.linspace(ci[i], ci[i+1],
        folds+1, endpoint=True),0).astype(int)
        for i in xrange(len(cc))])
    # start_stop is a ny x kfolds + 1 array with start and stop indices
    # for each class and fold
    result = []
    for f in xrange(folds):
        temp = [rarray[start_stop[i,f] : start_stop[i, f+1]]
                for i in xrange(len(cc))]
        result.append(list(chain.from_iterable(temp)))
    return result 

def get_conf_matrix(true, pred, class_ratios=None):
    '''
    Get a confusion matrix
    
    Input:
    ------
    true - the true labels
    pred - the predicted labels
    class_ratios - None or numpy array
                 - the actual ratio of classes; if None, it is
                 assumed that the actual class ratio equals the class
                 ratio during cross-validation. If this is not true
                 the actual class ratio can be given a numpy array
                 of length (number of classes).
                 class_ratios[0] is the actual frequency of class 0
                 class_ratios[1] is the actual frequency of class 1
                 .
                 .
                 .
    
    Output:
    -------
    conf_matrix - shape ny x ny, where ny is the number of classes
                 the rows belong to the actual, the columns to the
                 predicted class: item ij is hence predicted as class j,
                 while it would have belonged to class i
                
    '''
    # let the smallest class label be 0
    s = _np.min([true, pred], None)
    true -= s
    pred -= s
    # find the maximum number of classes (classes are consecutive
    # integers)
    n = _np.max([true, pred], None) + 1
    conf_matrix = _np.bincount(n * (true) + (pred),
            minlength=n*n).reshape(n, n).T
    if class_ratios != None:
        (assert isinstance(class_ratios, _np.ndarray),
                'class_ratios must be None or 1d numpy array')
        (assert class_ratios.ndim==1, 
                'dimensionality of class_ratios must be 1')
        (assert len(class_ratios)==n,
                'length of class_ratios must match number of classes')
        conf_matrix = (conf_matrix.T / conf_matrix.sum(1).astype(float) * class_ratios).T
    return conf_matrix

class ClassELM:
    '''
    Class for Extreme Learning Machine Classification
    --------------------------------------------
    
    Input:
    ------
    L - (int) - dimensionality of the feature space (defaults to 1000)
    change_alg - (int) - number of samples to change from implementation
                         I to II
    kernel - (str) - any of: 'sigmoid' - Sigmoid function
                             - more functions not implemented yet
    --------------------------------------------
    
    use self.cv() for cross-validation
    use self.train() for training
    
    after cross-validation or training use
    
    self.classifiy()
    '''
    def __init__(self, L=1000, kernel='sigmoid'):
        if type(L) == int: self.L = L
        else:
            raise TypeError('L must by an integer, representing the' +
                            'number of hidden neurons.')
        if kernel == 'sigmoid':
            self.kernel = _sigmoid
        else:
            raise Exception('Only kernel \'sigmoid\' is implemented' +
                            'at the current stage.')
        self._w = False # set weights to False
        self._pseudoy = False # set pseudo-output to false
        return

    def cv(self, data, labels, method='ssk_cv', C_array=None, folds=3,
            precision_func='accuracy', scale = True, weights=True,
            class_ratios=None, mem_size=512, verbose = True):
        '''
        Perform Cross-Validation of Extreme Learning Machine parameter C
        
        Input:
        ------
        data - numpy array - shape (n x p) with n being sample number
               and p being number of features
        labels - numpy array - shape (n) with the class labels
                 0,1,...,ny-2,ny-1, where ny is the number of classes
        method - string - cross-validation method
                        - 'ssk_cv' - shuffled stratified k-folds
                          cross-validation
        C_array - numpy array - default is None - the C's which are
                                cross-validated
                              - if None from 2**(-25), 2**(-24), ...,
                                             2**(24), 2**(25)
        folds - integer - default 3 - number of folds
        precision_func - string or function - standard is 'accuray' -
                         Measure of performance
                       - as string implemented: 'accuracy' -
                                                  proportion of
                                                  correctly classified
                                                  to total number of
                                                  samples
                                                'G_mean' - geometric
                                                  mean of per-class
                                                  accuracies
                                                'Matthews' - Matthews
                                                  Correlation
                                                  Coefficient - Only
                                                  for binary
                                                  classification
                                                'PPV' - Positive
                                                  Predictive Value -
                                                  Only for binary
                                                  classification
                       - if function: with confusion matrix as single
                         input and float (0,1) as single output
        scale - bool (True | False) - whether data should be scaled to
                                      range (-1,1)    
        weights - can be: - bool (True | False): - standard is True
                                                   if True, data is
                                                   re-weighted to a
                                                   class ratio of 1.0
                                                   if False, data is not
                                                   re-weighted
                          - float in half-open interval [0,1)
                                - data is re-weighted such that the
                                  minority / majority ratio is this
                                  float
                                - minority classes are the ones having
                                  less members than on average, majority
                                  classes have more than average
                                - Zong et al. proposed to use the golden
                                  ratio (approx. 0.618 -
                                  scipy.Constants.golden) as a good
                                  value
                          - numpy array with weights for each sorted
                            unique class in labels, each class weight is
                            expected to be in half-open interval [0,1) -
                            each class is "down-weighed" by this float
                            value, where the result depends on the ratio
                            this values to each other
        class_ratios - None or numpy array
                     - the actual ratio of classes; if None, it is
                     assumed that the actual class ratio equals the class
                     ratio during cross-validation. If this is not true
                     the actual class ratio can be given a numpy array
                     of length (number of classes).
                     class_ratios[0] is the actual frequency of class 0
                     class_ratios[1] is the actual frequency of class 1
                     .
                     .
                     .
        mem_size - int or float - default 512 - calculation is done in
                                  batches of this size in Mb
        
        Output:
        -------
        class instance of ClassELM        
        '''
        from itertools import chain
        if C_array == None:
            C_array = 2**_np.arange(-25,25,1).astype(float)
        if method == 'ssk_cv':
            get_folds = ssk_cv
        else:
            raise NotImplementedError('Only shuffled stratified' +
                    'k-fold cross-validation (\'ssk_cv\') is' +
                    'implemented.')
        if type(precision_func) == str:
            if precision_func == 'accuracy':
                precision_func = accuracy
            elif precision_func == 'G_mean':
                precision_func = G_mean
            elif precision_func == 'Matthews':
                precision_func = Matthews
            elif precision_func == 'PPV':
                precision_func = PPV
            else:
                raise Exception('The function \'%s\' is not' +
                'implemented (yet?).' % (precision_func))
        # check and get weights
        self._get_w(weights=weights, labels = labels)
        # create pseudo_output for each label    
        self._get_pseudoy(labels = labels)
        #scale the dataset with the complete (!) training
        # set - this is why scaling is set to False
        # in the train and classifiy functions later
        self.min = _np.min(data, 0)
        self.ptp = _np.ptp(data, 0)
        if scale:
            data = 2 * (data - self.min) / self.ptp - 1
        # cut data into folds
        partitions = get_folds(data, labels, folds)
        # partitions is a list with sublists of indices of each fold
        print 'Running Cross-Validation'
        result = _np.ones(C_array.shape, float)
        for n,C in enumerate(C_array):
            for k in xrange(folds):
                test = partitions[k] # fold k is used to test
                train = list(chain.from_iterable(partitions[:k] +
                    partitions[k+1:]))
                # the other (k-1) folds are used to train the network
                #in each training instance new random initialization
                # parameters are created
                self.train(data[train], labels[train], C=C,
                        mem_size = mem_size, scale=False,
                        weights=weights) # -> the weigh argument to that
                #method is no ignored since self._w already was
                # initialized get the estimated labels
                est_labels = self.classify(data[test], scale = False)
                conf_matrix = get_conf_matrix(labels[test], est_labels, class_ratios=class_ratios)
                result[n] = result[n] * precision_func(conf_matrix)
            if ((verbose) and (n % 1 == 0)):
                print 'Finished %d of %d Cross-Validations.' % (n+1,
                        len(C_array))
        result = result**(1./folds)
        # fix C as the C with the best cv-result
        try:
            C = C_array[_np.nanargmax(result)]
        except:
            C = C_array[0]
        # now train the network with the final C
        print 'Cross-Validation finished, Training final network'
        self.train(data, labels, C=C, mem_size=mem_size, scale = False,
                weights=self._w)
        print 'Network trained'
        return result
        
    def train(self, data, labels, C, scale=True, weights = True,
            mem_size=512):
        '''
        Train the ELM Classifier
        -----------------------
        
        Input:
        ------
        data - (numpy array) - shape n x p
                               n - number of observations
                               p - number of dimensions
               if data.ndim > 2, the array is reshaped as (n,-1)
        labels - array with integer labels
        C - regularization parameter    
        scale - bool (True | False) - standard is True
                                    - switch, if the features of the
                                      dataset should be scaled
                                      to the interval (-1,1)
        weights - can be: - bool (True | False): - standard is True
                                                   if True, data is re
                                                   weighted to a class
                                                   ratio of 1.0 if
                                                   False, data is not
                                                   re-weighted
                          - float in half-open interval [0,1)
                                - data is re-weighted such that the
                                  minority / majority ratio is this
                                  float
                                - minority classes are the ones having
                                  less members than on average, majority
                                  classes have more than average
                                - Zong et al. proposed to use the golden
                                ratio (approx. 0.618 -
                                scipy.Constants.golden) as a good value
                          - numpy array with weights for each sorted
                          unique class in labels, each class weight is
                          expected to be in half-open interval [0,1)
        mem_size - number - memory size of temporary array in Mb -
                            defaulte to 512    
        
        Output:
        -------
        No user ouput (Weights are generated and stored in the Class as
        self._beta) self.istrained is set to True
        '''
        # reshape data
        data = data.reshape(data.shape[0],-1)
        n,p = data.shape
        self.n = n
        self.p = p
        self.C = C
        # initialize parameters
        a, b = _get_parameters(self.L, self.p)
        self.a = a
        self.b = b
        # transform labels
        if _np.all(self._w == False): # initialized as False in the
            #class definition
            # if weights are not already fixed, they should be
            # determined here if weighing should be performed
            self._get_w(weights=weights, labels = labels)
        if not _np.any(self._pseudoy): # if the pseudo-output has not
            # been created before, do it now
            self._get_pseudoy(labels)
        if scale:
            # normalize the dataset to range (-1,1)
            self.min = _np.min(data,0)
            self.ptp = _np.ptp(data, 0)
            data = 2 * (data - self.min) / self.ptp - 1
        if n <= self.L:
            #self._algI(data, labels, mem_size=mem_size)
            # algI is not implemented yet so up to now use 
            # algorithm I in any case
            self._algII(data, labels, mem_size=mem_size)
        else:
            self._algII(data, labels, mem_size=mem_size)
        self.istrained = True
        return
    
    def _get_w(self, weights, labels):
        '''
        This method checks the input argument 'weights'
        and broadcasts it to the class parameter self._w
        '''
        if type(weights) == bool:
            if weights == True:
                self.weigh = True
                w = 1.
            else: self.weigh = False
        else:
            # check if weights is a number
            from numbers import Number
            if isinstance(weights, Number):
                if ((weights > 0) and (weights <= 1.0)):
                    self.weigh = True
                    w = float(weights)
                else:
                    raise Exception('weights should be either a' +
                    'Boolean, a numpy array or a number in the' +
                    'half-open interval [0,1)')
            elif type(weights) == _np.ndarray:
                if ((weights.size == labels.max() + 1) and
                     _np.all(weights > 0) and _np.all(weights <= 1.0)):
                    self.weigh = True
                    self._w = weights
                else:
                    raise TypeError('weights should be either a' +
                    'Boolean, a numpy array or a number in the' +
                    'half-open interval [0,1)')
            else:
                raise TypeError('weights should be either a Boolean,' +
                'a numpy array or a number in the half-open'
                'interval [0,1)')
        if self.weigh and not _np.any(self._w):
            # if re-weighing of imbalances should occur, then find
            #the minority and majority classes
            # find number of classes per label
            n_per_class = _np.bincount(labels)
            # assign w / n_per_class to the minority classes and
            # 1.0 / n_per_class to the majority classes
            self._w = _np.where(n_per_class < n_per_class.mean(), w /
                    n_per_class, 1.0 / n_per_class)
        return
    
    def _get_pseudoy(self, labels):
        ul = _np.unique(labels)
        if len(ul) == 1:
            self._pseudoy = _np.array([1,1]).astype(int)
            self.m = 1
        elif len(ul) == 2:
            self._pseudoy = _np.array([-1,1]).astype(int)
            self.m = 1
        else:
            self._pseudoy = _np.eye(len(ul)).astype(int)
            self.m = len(ul)   
        return
    
    def _algII(self, data, labels, mem_size):
        '''
        Train ELM with algorithm II
        i.e. formula (38) in:
        Huang et al.: Extreme Learning Machine for Regression and
        Multiclass Classification
        IEEE Transactions of Systems, Man, and Cybernetics - Part B:
        Cybernetics, Vol 42, No. 2, April 2012
        '''
        ###
        # split data into batches of maximum size=mem_size
        n = data.shape[0]
        data_bytes = data.nbytes / data.size
        batch_len = int(mem_size * 1024.0**2 / data_bytes / self.L)
        num_batches = int(_np.ceil(n / float(batch_len)))
        borders = _np.linspace(0,n,num_batches+1,
                endpoint=True).astype(int)
        # initialze result array
        HTH = _np.zeros([self.L,self.L], dtype=data.dtype)
        if self.m == 1:
            HTT = _np.zeros(self.L, HTH.dtype)
        else:
            HTT = _np.zeros((self.L, self.m), HTH.dtype)
        # run in batches
        for k in xrange(num_batches):        
            temp = self._get_HTH_HTT(data =
                    data[borders[k]:borders[k+1]],
                    labels=labels[borders[k]:borders[k+1]])
            HTH += temp[0]
            if self.m == 1:
                HTT += temp[1].reshape(self.L)
            else:
                HTT += temp[1]
        try: # solution might be invalid due to singular matrix
            self._beta = _linalg.solve(_np.diag(1./self.C *
                _np.ones(self.L)) + HTH, HTT, sym_pos=True,
                check_finite=False)
        except:
            try:
                #try least squares solution
                self._beta = _linalg.lstsq(_np.diag(1./self.C *
                    _np.ones(self.L)) + HTH, HTT, check_finite=False)[0]
            except:
                raise Exception('This did not work')
        return
    
    def classify(self, data, mem_size = 512, scale=True):
        '''
        Classify a dataset.
        
        Input:
        ------
        data - numpy array, shape N x p, where N is number of items, p
               is number of features
        mem_size - number - memory size of temporary array in Mb -
                   defaulte to 512
        scale - bool (True | False) - if the input should be scaled
                by the network with parameters obtained during training
        Output:
        -------
        labels - the predicted class labels: 0, 1, ..., ny-2, ny-1,
                 where ny is the total number of classes
        ----------------------------------------------------------------
        Internally the method _run() is used
        
        '''
        if self.istrained == False:
            raise Exception('Network is not trained.')
        if scale:
            data = 2 * (data - self.min) / self.ptp - 1
        out = self._run(data, mem_size = mem_size)
        if self.m == 1:
            return _np.sign(out).clip(0,1).astype(int)
        else:
            return _np.nanargmax(out, -1).astype(int)
    
    def _run(self, data, mem_size=512):
        '''
        Internal function! Not for end user! Use the method classify!
        -------------------------------------------------------------
        Get the responses of the neuron when exposed to input data
        '''
        n = data.shape[0]
        data_bytes = data.nbytes / data.size
        batch_len = int(mem_size * 1024.0**2 / data_bytes / self.L)
        num_batches = int(_np.ceil(n / float(batch_len)))
        borders = _np.linspace(0,n,num_batches+1,
                endpoint=True).astype(int)
        if self.m > 1:
            out = _np.empty((n, self.m), float)
        else:
            out = _np.empty((n), float)
        for k in xrange(len(borders)-1):
            out[borders[k]:borders[k+1]] = self.kernel(
                    data[borders[k]:borders[k+1]],
                    self.a, self.b).T.dot(self._beta)
        return out
   
    def _get_HTH_HTT(self, data, labels):
        '''
        Internal function! Not for end user! 
        ------------------------------------
        Needed for algorithm II
        '''
        temp = self.kernel(data, self.a, self.b)
        if self.weigh:
            HTH = _dot(temp * self._w[labels], temp.T)
            HTT = _np.dot(temp  * self._w[labels],
                    self._pseudoy[labels])
        else:
            HTH = _dot(temp, temp.T)
            HTT = _np.dot(temp, self._pseudoy[labels])
        return HTH, HTT    

def _get_parameters(L, p):
    '''
    Internal function! Not for end user! 
    ------------------------------------
    Initialize random parameters a and b
    a - neuron center in interval (-1,1)
    b - neuron width in interval (0,1)
    '''
    a = _np.random.uniform(low=-1.0, high=1.0, size=(L,p))
    b = _np.random.uniform(low = 0.0, high=1.0, size=L)
    return a, b

def _algI(data, a, b, kernel, C, labels):
    '''
    Train ELM with algorithm I
    i.e. formula (32) in:
    Huang et al.: Extreme Learning Machine for Regression and Multiclass
    Classification
    IEEE Transactions of Systems, Man, and Cybernetics - Part B:
    Cybernetics, Vol 42, No. 2, April 2012
    '''
    #######################
    # not implemented yet #
    #######################
    raise NotImplementedError('Algorithm I is not implemented yet!')

def _sigmoid(data, a, b):
    '''
    Calculate response of neurons with sigmoid kernel
    and pre-generated parameters a and b.
    
    Output shape is L x n, where L is number of neurons
    and n is number of input samples
    '''
    if a.ndim > 1: b = b[:,None]
    return 1. / (1 + _np.exp(-1*(_np.dot(a,data.T) + b)))
