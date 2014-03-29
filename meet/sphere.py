'''
Spherical spline interpolation and CSD

Submodule of the Modular EEg Toolkit - MEET for Python.

Algorithm from  Perrin et al., Electroenceph Clin Neurophysiol 1989,72:184-187, Corrigenda in 1990, 76: 565-566

While the implementation was done independently, the code was tested using the sample data of the CSD Toolbox for Matlab
(http://psychophysiology.cpmc.columbia.edu/Software/CSDtoolbox/)
and results are the same. Thanks a lot th Juergen Kayer for sharing his work there.

Example:
--------
For Plotting a scalp map:

>>> import numpy as _np
>>> coords = getStandardCoordinates(['Fp1', 'Fp2', 'C1', 'C2', 'C3', 'C4', 'P2', 'P4'])
>>> coords = projectCoordsOnSphere(coords) # be sure that these coords are on the surface of a unit sphere
>>> data = _np.random.random(coords.shape[0]) # just create a random vector for testing
>>> X, Y, Z = potMap(coords, data) # interpolate using spherical splines

For Calculating current source densities:

>>> import numpy as _np
>>> coords = getStandardCoordinates(['Fp1', 'Fp2', 'C1', 'C2', 'C3', 'C4', 'P2', 'P4'])
>>> coords = projectCoordsOnSphere(coords) # be sure that these coords are on the surface of a unit sphere
>>> data = _np.random.random([coords.shape[0], 1000]) # just create a random vector for testing
>>> CSD = calcCSD(coords, data) # get CSD using spherical splines - output is in data-unit/m**2

Author:
-------
Gunnar Waterstraat
gunnar[dot]waterstraat[at]charite.de
'''

from __future__ import division
from . import _np
from . import _path
from . import _packdir
from numpy.polynomial.legendre import legval as _legval

def getStandardCoordinates(elecnames,fname='standard'):
    """
    Read (cartesian) Electrode Coordinates from tab-seperated text file

    The standard is plotting_1005.txt obtained from:
    http://robertoostenveld.nl/electrode/plotting_1005.txt

    Thanks to Robert Oostenveld for sharing these files!!!

    Input:
    ------
    -- elecnames - iterable - list of Electrode names
    -- fname - str - filename from which positions should be read
                   - this is a tab delimeed file with the UPPERCASE electrode names in first column
                     x,y,z in subsequent columns
    Output:
    -------
    -- coords - 2D array containing the cartesian coordinates in rows
                    - 1st column: x
                    - 2nd column: y
                    - 3rd column: z

    Example:
    --------
    >>> getStandardCoordinates(['fc3', 'FC1', 'xx']) # 'xx' is not a valid electrode
    array([[-0.6638    ,  0.3610691 ,  0.6545    ],
           [-0.3581    ,  0.37682936,  0.8532    ],
           [        nan,         nan,         nan]])
    """
    if fname == 'standard':
        fname = _path.join(_packdir, 'plotting_1005.txt')
    #make all elecnames uppercase
    elecnames = [e.upper() for e in elecnames]
    from csv import reader
    filereader = reader(open(fname), delimiter='\t')
    coords = dict([(row[0].upper(),row[1:4]) for row in filereader])
    #get origin of the used coordinate system
    x0 = _np.array([coords['T7'][0],coords['T8'][0]],dtype=float).mean()
    y0 = _np.array([coords['OZ'][1],coords['FPZ'][1]],dtype=float).mean()
    z0 = _np.array([coords['T8'][2],coords['T7'][2]],dtype=float).mean()
    # move the electrodes accordingly
    for key in coords.iterkeys():
        coords[key] = _np.array(coords[key],dtype=float)
        coords[key][1] -= x0
        coords[key][1] -= y0
        coords[key][2] -= z0
    XA = _np.array(coords['T8']).astype(float) #electrode with (1,0,0)
    YA = -1 * _np.array(coords['OZ']).astype(float) #electrode with (0,1,0)
    ZA = _np.array(coords['CZ']).astype(float) #electrode with (0,0,1)
    TransMat = _np.matrix(_np.array([XA, YA, ZA], dtype=float).T)
    coords_result = []
    for item in elecnames:
        if item in coords:
            temp = list(_np.array(TransMat.I * _np.matrix(_np.array(coords[item], \
            dtype=float)).T).flatten())
            coords_result.append(temp)
        else: coords_result.append(_np.array([_np.nan,_np.nan,_np.nan])) # if not in file
    coords = _np.array(coords_result,dtype=float)
    return coords

def getChannelNames(fname):
    """
    Read Names of Electrodes from text file

    Input:
    ------
    -- fname - str - tab-delimeted textfile containing electrode number and electrode-names
                     example:
                     1  C1
                     2  C3
    output:
    -------
    -- elecnames -  List of electrode names in ascending order as determined by the electrode numbers in fname

    Example:
    --------
    >>> getChannelNames(_path.join(_path.join(_packdir, 'test_data'), 'elecnames.txt'))
    ['C1', 'C3']
    """
    from csv import reader
    filereader = reader(open(fname), delimiter='\t', quotechar='"')
    elecnames = dict([(row[0].upper(),int(row[1])) for row in filereader])
    from operator import itemgetter
    elecnames = sorted(elecnames.iteritems(), key=itemgetter(1))
    return [item[0] for item in elecnames]

def projectCoordsOnSphere(coords):
    """
    Get the Coords with same azimuth and altitude on on a spherical surface
    with center 0 and radius 1

    Input:
    ------
    --coords - 3D array containing the points in rows
               - 1st column: x
               - 2nd column: y
               - 3rd column: z
    Output:
    -------
    -- out_coords - 3D array containing the coordinates in rows

    Examples:
    --------
    >>> coords = _np.array([[0,0,1], [0,1,0], [1,1,1]]) # the last one is not on a spherical surface with radius 1
    >>> projectCoordsOnSphere(coords)
    array([[ 0.        ,  0.        ,  1.        ],
           [ 0.        ,  1.        ,  0.        ],
           [ 0.57735027,  0.57735027,  0.57735027]])
    """
    t = _np.sqrt(1./ ((coords**2).sum(axis=-1)))
    out_coords = t[:,_np.newaxis]*coords
    return out_coords

def meshCircle(d_samples=100):
    """
    Calculate a meshgrid containing the points of a circle (center (0,0),
    radius 1)
    
    Input:
    ------
    -- d_samples - number of points along the diameter

    Output:
    -------
    -- coords - masked 2D array containing the coordinates in rows
              - 1st column: x
              - 2nd column: y
       (everything otside a unit sphere is masked)
    """
    coords = []
    x = y = _np.linspace(-1, 1, d_samples, endpoint=True)
    X, Y = _np.meshgrid(x,y)
    #X = X.ravel()
    #Y = Y.ravel()
    #remove points outside the circle
    #inlier  = X**2 + Y**2 <= 1
    #return _np.array([X[inlier], Y[inlier]]).T
    X = _np.ma.masked_where(X**2 + Y**2 > 1, X)
    Y = _np.ma.masked_where(X**2 + Y**2 > 1, Y)
    return _np.ma.column_stack([_np.ma.ravel(X), _np.ma.ravel(Y)])

def meshCircleOnSphere(coords):
    """
    Calculate a 3D meshgrid on a sphere from a 2D meshgrid on a circle
    with center (0,0) and radius 1

    Input:
    ------
    -- coords - 2D array containing the coordinates in rows
              - 1st column: x
              - 2nd column: y
    Output:
    ------
    -- sphere_coords - masked 3D array containing the coordinates in rows
                     - 1st column: x
                     - 2nd column: y
                     - 3rd column: z
     (all coordinates outside a unit sphere are masked)
    """
    theta = _np.ma.sqrt((coords**2).sum(1))
    k = _np.ones_like(theta)
    k[theta != 0] = _np.sin(theta[theta != 0]) / theta[theta != 0]
    x  = (k*coords[:,0])
    y = (k*coords[:,1])
    z  = _np.ma.sqrt(1 - x**2 - y**2)
    return _np.ma.column_stack([x,y,z])

def meshSphereOnCircle(sphere_coords):
    """
    Calculate a 2D meshgrid on a circle from a 3D meshgrid on a sphere

    Input:
    ------
    -- sphere_coords - 2D array containing the coordinates in rows
              - 1st column: x
              - 2nd column: y
              - 3rd column: z
    Output:
    -------
    -- coords - 2D array containing the coordinates in rows
                     - 1st column: x
                     - 2nd column: y

    Example:
    --------
    >>> coords_2d = meshCircle(4) # mesh a circle with 4 points on the diagonal
    >>> coords_3d = meshCircleOnSphere(coords_2d) # project that mesh on a sphere
    >>> _np.all(_np.abs((meshSphereOnCircle(coords_3d) - coords_2d) / coords_2d) < 1E-10)
    True
    """
    theta = _np.arcsin(_np.sqrt(1 - sphere_coords[:,2]**2))
    k = _np.ones_like(theta)
    k[theta != 0] = _np.sin(theta[theta != 0]) / theta[theta != 0]
    return sphere_coords[:,:2] / k[:,_np.newaxis]

def _getGH(coords1, coords2, m=4, n=7, which='G'):
    '''
    Internal Function for computation of CSD
    and Interpolation

    Input
    ------
    -- coords1
    -- coords2
    -- which - str - 'G' or 'H'

    Output:
    -------
    -- G (if which == 'G')
    -- H (if which == 'H')
    '''
    coords1 = coords1.astype(float)
    coords2 = coords2.astype(float)
    # cosine of angle is the inner product divided by the product of 2-norms
    cos_angle = coords1.dot(coords2.T) / ((coords1**2).sum(1)[:,_np.newaxis] * (coords2**2).sum(1)[_np.newaxis])
    N = _np.arange(1,n+1,1, dtype=float)
    if which == 'G':
        #evaluate Legendre polynomial
        c_g = ( 2*N + 1) / (N**2 + N)**m #coefficients for g
        #start with 1st (not 0st) polynomial
        c_g = _np.hstack([[0], c_g])
        return _legval(cos_angle, c_g) / (4 * _np.pi) # this is G in the Perrin publication
    elif which == 'H':
        c_h = (-2*N - 1) / (N**2 + N)**(m-1) #coefficients for h
        #start with 1st (not 0st) polynomial
        c_h = _np.hstack([[0], c_h])
        return -1 * _legval(cos_angle, c_h) / (4 * _np.pi) # this is H in the Perrin publication
    else:
        raise ValueError('which must be G or H')

def _sphereSpline(data, G_hh, G_hw=None, H_hw=None, smooth=0, type='Interpolation', buffersize=64):
    """
    Internal Function for computation of CSD
    and Interpolation

    -- data - 2dim with channels x datapoints
    -- G_hh - G between real electrodes (hh = have-have)
    -- G_hw - G between real and wanted electrodes (hw = have-wanted) - needed for interpolation
    -- H_hw - H between real and wanted electrodes (hw = have-wanted) - needed for CSD
    -- smooth - float - set to zero if no smoothing should be applied
    """
    # Prepare the relevant matrices
    # make it 2d if not already so
    data = data.reshape(data.shape[0],-1)
    p, n = data.shape
    # get number of batches
    num_batches = int(_np.ceil((data.nbytes / n ) / float(buffersize*1024**2)))
    delims = _np.linspace(0, n, num_batches+1, endpoint=True)
    #add smothing parameter to G
    G_hh[range(p), range(p)] = G_hh[range(p), range(p)] + smooth
    #change g to solve matrix system
    G_hh = _np.ma.hstack([_np.ones([G_hh.shape[0],1], G_hh.dtype), G_hh])
    G_hh = _np.ma.vstack([_np.ones([1,G_hh.shape[1]], G_hh.dtype), G_hh])
    G_hh[0,0] = 0
    out = []
    for i in xrange(num_batches):
        #c0 is first row, the other cs are in the following rows, each column is for one datapoint of n
        C = _np.linalg.solve(G_hh, _np.vstack([_np.ones(delims[i+1]-delims[i], data.dtype), data[:,delims[i]:delims[i+1]]]))
        if type == 'Interpolation':
            out.append(C[0,:] + _np.ma.dot(G_hw.T, C[1:], strict=True))
        elif type == 'CSD':
            out.append(_np.ma.dot(H_hw.T, C[1:], strict=True))
        else: raise ValueError('Type must be Interpolation or CSD')
    result = _np.ma.hstack(out)
    if _np.all(result.mask == False):
        return result.data
    else:
        return result

def potMap(RealCoords, data, diameter_samples=200, n=(7,7), m=4, smooth=0):
    """
    Get a scalp map of potentials interpolated on a sphere.

    Input:
    ------
    -- RealCoords - array of shape channels x 3 - (x,y,z) cartesian coordinates
                                                  of physical electrodes on a
                                                  sphere with center (0,0,0) and
                                                  radius 1
    -- data - array of shape channels x datapoints - containing values at
                                                     electrodes in same order as
                                                     in 'RealCoords'
    -- diameter_samples - int - number of points along the diameter of the scalp
                                to interpolate
    -- n - tuple of ints - how many terms of the Legendre-Polynomial should be calculated,
                           defaults to (7,7)
    -- m - int - order of the spherical spline interpolation, defaults to 4
    -- smooth - float - amount of smoothing, defaults to 0
    
    Output:
    -------
    -- X, Y, Z - grid containing the X and Y coordinates and the interpolated values
                 where everything outside the unit circle is masked.
    """
    InterpCoords_2d = meshCircle(d_samples=diameter_samples)
    InterpCoords_3d = meshCircleOnSphere(InterpCoords_2d)
    RealCoords = projectCoordsOnSphere(RealCoords) # be sure that these coords are on the surface of a unit sphere
    G_hh = _getGH(RealCoords, RealCoords, m=m, n=n[0], which='G')
    G_hw = _getGH(RealCoords, InterpCoords_3d, m=m, n=n[1], which='G')
    pot = _sphereSpline(data, G_hh=G_hh, G_hw=G_hw, smooth=smooth, type='Interpolation')
    return InterpCoords_2d[:,0].reshape(diameter_samples, diameter_samples), InterpCoords_2d[:,1].reshape(diameter_samples, diameter_samples), pot.reshape(diameter_samples, diameter_samples)

def csdMap(RealCoords, data, diameter_samples=200, n=(7,20), m=4, smooth=1E-5):
    """
    Get a scalp map of CSDs calculated from spherical splines.

    Input:
    ------
    -- RealCoords - array of shape channels x 3 - (x,y,z) cartesian coordinates
                                                  of physical electrodes on a
                                                  sphere with center (0,0,0) and
                                                  radius 1
    -- data - array of shape channels x datapoints - containing values at
                                                     electrodes in same order as
                                                     in 'RealCoords'
    -- diameter_samples - int - number of points along the diameter of the scalp
                                to interpolate
    -- n - tuple of ints - how many terms of the Legendre-Polynomial should be calculated,
                           defaults to (7,20)
    -- m - int - order of the spherical spline interpolation, defaults to 4
    -- smooth - float - amount of smoothing, defaults to 1E-5
    
    Output:
    -------
    -- X, Y, Z - grid containing the X and Y coordinates and the interpolated values
                 where everything outside the unit circle is masked.
                 the unit of z is data_unit / m**2
    """
    InterpCoords_2d = meshCircle(d_samples=diameter_samples)
    InterpCoords_3d = meshCircleOnSphere(InterpCoords_2d)
    RealCoords = projectCoordsOnSphere(RealCoords) # be sure that these coords are on the surface of a unit sphere
    G_hh = _getGH(RealCoords, RealCoords, m=m, n=n[0], which='G')
    H_hw = _getGH(RealCoords, InterpCoords_3d, m=m, n=n[1], which='H')
    pot = _sphereSpline(data, G_hh=G_hh, H_hw=H_hw, smooth=smooth, type='CSD')
    return InterpCoords_2d[:,0].reshape(diameter_samples, diameter_samples), InterpCoords_2d[:,1].reshape(diameter_samples, diameter_samples), pot.reshape(diameter_samples, diameter_samples)

def calcCSD(Coords, data, n=(7,20), m=4, smooth=1E-5, buffersize=64):
    """
    Calculate current source densities at the same electrode posisionts as in the
    original data set
    
    Input:
    ------
    -- Coords - 2d array - channels x (x,y,z) cartesian coordinates of physical electrodes
    -- data - array containing values at electrodes in same order as in 'Coords'
        -- 1st dim: channels
        -- 2nd dim: datapoints
    -- n - tuple of ints - How many terms of the Legendre-Polynomial should be calculated,
                           defaults to (7, 20) - 7 for the first and 20 for the 2nd polynomial
    -- m - order of the spherical spline interpolation, defaults to 4
    -- smooth - float - amount of smoothing, defaults to 1E-5
    -- buffersize - float - determine the size of a buffer in MB to do the calculations

    Output:
    -------
    -- pot - array containing CSD-values in data-unit / m**2

    Example:
    --------
    >>> Coords = getStandardCoordinates(['C1', 'C3'])
    >>> Coords = projectCoordsOnSphere(Coords) # be sure that the points are on a sphere
    >>> data = _np.arange(6, dtype=float).reshape(2,-1)
    >>> calcCSD(Coords, data)
    array([[-3.5474569 , -3.5474569 , -3.5474569 ],
           [ 3.60884286,  3.60884286,  3.60884286]])
    """
    Coords = projectCoordsOnSphere(Coords) # be sure that these coords are on the surface of a unit sphere
    G_hh = _getGH(Coords, Coords, n=n[0], m=m, which='G')
    H_hw = _getGH(Coords, Coords, n=n[1], m=m, which='H')
    pot = _sphereSpline(data, G_hh = G_hh, H_hw = H_hw, buffersize=buffersize, type='CSD', smooth=smooth)
    return pot

def smoothSP(Coords, data, n=(7,7), m=4, smooth=1E-5, buffersize=64):
    """
    Calculate spatially smoothed potential values at the same electrode positions as in the
    original data set
    
    Input:
    ------
    -- Coords - 2d array - channels x (x,y,z) cartesian coordinates of physical electrodes
    -- data - array containing values at electrodes in same order as in 'Coords'
        -- 1st dim: channels
        -- 2nd dim: datapoints
    -- n - tuple of ints - How many terms of the Legendre-Polynomial should be calculated,
                           defaults to (7, 7) - 7 for the first and 7 for the 2nd polynomial
    -- m - order of the spherical spline interpolation, defaults to 4
    -- smooth - float - amount of smoothing, defaults to 1E-5
    -- buffersize - float - determine the size of a buffer in MB to do the calculations

    Output:
    -------
    -- pot - array containing the spatially smoothed potentials
    
    Example:
    --------
    >>> Coords = getStandardCoordinates(['C1', 'C3'])
    >>> Coords = projectCoordsOnSphere(Coords) # be sure that the points are on a sphere
    >>> data = _np.arange(6, dtype=float).reshape(2,-1)
    >>> smoothSP(Coords, data)
    array([[ 0.01220571,  1.01220571,  2.01220571],
           [ 2.98778429,  3.98778429,  4.98778429]])
    """
    Coords = projectCoordsOnSphere(Coords) # be sure that these coords are on the surface of a unit sphere
    G_hh = _getGH(Coords, Coords, n=n[0], m=m, which='G')
    G_hw = _getGH(Coords, Coords, n=n[1], m=m, which='G')
    pot = _sphereSpline(data, G_hh = G_hh, G_hw = G_hw, buffersize=buffersize, type='Interpolation', smooth=smooth)
    return pot
