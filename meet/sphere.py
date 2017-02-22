'''
Spherical spline interpolation and CSD

Submodule of the Modular EEg Toolkit - MEET for Python.

Algorithm from  Perrin et al., Electroenceph Clin Neurophysiol 1989,
72:184-187, Corrigenda in 1990, 76: 565-566

While the implementation was done independently, the code was tested
using the sample data of the CSD Toolbox for Matlab
(http://psychophysiology.cpmc.columbia.edu/Software/CSDtoolbox/)
and results are the same. Thanks a lot to Juergen Kayer for sharing his
work there.

Example:
--------
For Plotting a scalp map:

>>> import numpy as _np
>>> coords = getStandardCoordinates(['Fp1', 'Fp2', 'C1', 'C2', 'C3', \
        'C4', 'P2', 'P4'])
>>> coords = projectCoordsOnSphere(coords) # be sure that these coords \
        are on the surface of a unit sphere
>>> data = _np.random.random(coords.shape[0]) # just create a random \
        vector for testing
>>> X, Y, Z = potMap(coords, data) # interpolate using spherical splines

For Calculating current source densities:

>>> import numpy as _np
>>> coords = getStandardCoordinates(['Fp1', 'Fp2', 'C1', 'C2', 'C3', \
        'C4', 'P2', 'P4'])
>>> coords = projectCoordsOnSphere(coords) # be sure that these coords \
        are on the surface of a unit sphere
>>> data = _np.random.random([coords.shape[0], 1000]) # just create a \
        random vector for testing
>>> CSD = calcCSD(coords, data) # get CSD using spherical splines - \
        output is in data-unit/m**2

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
from scipy.spatial import ConvexHull as _hull
from matplotlib import path as _mpath

# ECOD is used to solve linear equations
from ECOD import ECOD_LS as _solve

from numpy.linalg import cond as _cond

def addHead(ax, lw=2.0, ec='k', **kwargs):
    """
    Add a path representing a a head consisting of
    a circle with center (0,0) and r = 1 to the axis.

    Input:
    ------
    --ax - a matplotlib axes instance into which the head will be drawn
    --lw - linewidth - defaults to 2.0
    --ec - edgecolor - defaults to black ('k')

    Output:
    -------
    -- patch - the constructed circle

    Notes:
    ------
    Kwargs are matplotlib patches kwargs and are passed to
    matplotlib.patches.PathPatch

    Parts of the code (for drawing the circle) are from:
    https://sourcegraph.com/github.com/matplotlib/matplotlib/symbols/python/lib/matplotlib/path/Path/circle
    """

    from matplotlib.patches import PathPatch as _PathPatch
    #1st create circle
    magic = 0.2652031
    sqrthalf = _np.sqrt(0.5)
    magic45 = _np.sqrt(magic**2 / 2.)
    cvertices =  [ [0.0, -1.0],
        
                 [magic, -1.0],
                 [sqrthalf-magic45, -sqrthalf-magic45],
                 [sqrthalf, -sqrthalf],

                 [sqrthalf+magic45, -sqrthalf+magic45],
                 [1.0, -magic],
                 [1.0, 0.0],

                 [1.0, magic],
                 [sqrthalf+magic45, sqrthalf-magic45],
                 [sqrthalf, sqrthalf],

                 [sqrthalf-magic45, sqrthalf+magic45],
                 [magic, 1.0],
                 [0.0, 1.0],

                 [-magic, 1.0],
                 [-sqrthalf+magic45, sqrthalf+magic45],
                 [-sqrthalf, sqrthalf],

                 [-sqrthalf-magic45, sqrthalf-magic45],
                 [-1.0, magic],
                 [-1.0, 0.0],

                 [-1.0, -magic],
                 [-sqrthalf-magic45, -sqrthalf+magic45],
                 [-sqrthalf, -sqrthalf],

                 [-sqrthalf+magic45, -sqrthalf-magic45],
                 [-magic, -1.0],
                 [0.0, -1.0],

                 [0.0, -1.0]]

    ccodes = [_mpath.Path.CURVE4] * 26
    ccodes[0] = _mpath.Path.MOVETO
    ccodes[-1] = _mpath.Path.CLOSEPOLY
    # now create ears
    rvertices = [[_np.sqrt(1-0.2**2), 0.2],

                 [1.2, 0],
                 [_np.sqrt(1-0.2**2), -0.2]]
    rcodes = [_mpath.Path.MOVETO,
              _mpath.Path.CURVE3,
              _mpath.Path.CURVE3]
    lvertices = [[-_np.sqrt(1-0.2**2), 0.2],

                 [-1.2, 0],
                 [-_np.sqrt(1-0.2**2), -0.2]]
    lcodes = [_mpath.Path.MOVETO,
              _mpath.Path.CURVE3,
              _mpath.Path.CURVE3]
    # now create nose
    nvertices = [[-0.1,_np.sqrt(1 - 0.1**2)],
                 [0,1.15],
                 [0.1, _np.sqrt(1 - 0.1**2)]]
    ncodes = [_mpath.Path.MOVETO,
              _mpath.Path.LINETO,
              _mpath.Path.LINETO]
    #construct complete path
    path = _mpath.Path(cvertices + rvertices + lvertices + nvertices,
                       ccodes + rcodes + lcodes + ncodes)
    #construct patch
    patch = _PathPatch(path, transform=ax.transData, fill=False, lw=lw, ec=ec, **kwargs)
    #add to axes
    ax.add_patch(patch)
    return patch

def getStandardCoordinates(elecnames,fname='standard'):
    """
    Read (cartesian) Electrode Coordinates from tab-seperated text file

    The standard is plotting_1005.txt obtained from:
    http://robertoostenveld.nl/electrode/plotting_1005.txt (however 1st
    and 2nd column were exchanged to get x,y,z order)

    Thanks to Robert Oostenveld for sharing these files!!!

    Input:
    ------
    -- elecnames - iterable - list of Electrode names
    -- fname - str - filename from which positions should be read
                   - this is a tab delimeed file with the UPPERCASE
                     electrode names in first column
                     x,y,z in subsequent columns
                     T7, T8, Oz, Fpz and Cz must be included!
    Output:
    -------
    -- coords - 2D array containing the cartesian coordinates in rows
                    - 1st column: x
                    - 2nd column: y
                    - 3rd column: z

    Example:
    --------
    >>> getStandardCoordinates(['fc3', 'FC1', 'xx']) # 'xx' is not a \
            valid electrode
    array([[-0.6638    ,  0.3610691 ,  0.6545    ],
           [-0.3581    ,  0.37682936,  0.8532    ],
           [        nan,         nan,         nan]])
    """
    if fname == 'standard':
        fname = _path.join(_packdir, 'plotting_1005.txt')
    #make all elecnames uppercase
    elecnames = [e.upper() for e in elecnames]
    from csv import reader
    filereader = reader(open(fname, 'r'), delimiter='\t')
    coords = dict([(row[0].upper(),row[1:4]) for row in filereader])
    #get origin of the used coordinate system
    x0 = _np.array([coords['T7'][0],coords['T8'][0]],dtype=float).mean()
    y0 = _np.array([coords['OZ'][1],coords['FPZ'][1]],
            dtype=float).mean()
    z0 = _np.array([coords['T8'][2],coords['T7'][2]],dtype=float).mean()
    # move the electrodes accordingly
    for key in coords.iterkeys():
        coords[key] = _np.array(coords[key],dtype=float)
        coords[key][0] -= x0
        coords[key][1] -= y0
        coords[key][2] -= z0
    #transform to put all electrodes to their standard place
    #electrode that should be at (1,0,0)
    XA = _np.array(coords['T8']).astype(float) 
    #electrode that should be at (0,1,0)
    YA = _np.array(coords['FPZ']).astype(float)
    #electrode that should be at (0,0,1)
    ZA = _np.array(coords['CZ']).astype(float)
    TransMat = _np.array([XA, YA, ZA], dtype=float)
    coords_result = []
    for item in elecnames:
        if item in coords:
            coords_result.append(
            _np.linalg.lstsq(TransMat.T, coords[item])[0])
        else:
            coords_result.append(
                _np.array([_np.nan,_np.nan,_np.nan])) # if not in file
    coords = _np.array(coords_result,dtype=float)
    return coords

def getChannelNames(fname):
    """
    Read Names of Electrodes from text file

    Input:
    ------
    -- fname - str - tab-delimeted textfile containing electrode number
                     and electrode-names
                     example:
                     1  C1
                     2  C3
    output:
    -------
    -- elecnames -  List of electrode names in ascending order as
                    determined by the electrode numbers in fname

    Example:
    --------
    >>> getChannelNames(_path.join(_path.join(_packdir, 'test_data'), \
            'elecnames.txt'))
    ['C1', 'C3']
    """
    from csv import reader
    filereader = reader(open(fname, 'r'), delimiter='\t', quotechar='"')
    elecnames = dict([(row[0].upper(),int(row[1])) 
        for row in filereader])
    from operator import itemgetter
    elecnames = sorted(elecnames.iteritems(), key=itemgetter(1))
    return [item[0] for item in elecnames]

def projectCoordsOnSphere(coords):
    """
    The input coordinates are projected onto a sphere with center
    (0,0,0) and radius 1. 
    For the input coordinates it is believed that they lie on a
    sphere with center (0,0,0) and radius r = x**2 + y**2 + Z**2.
    Subsequently this radius is scaled th 1, preserving altitude
    and azimuth of the original coordinates.

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
    >>> coords = _np.array([[0,0,1], [0,1,0], [1,1,1]]) # the last one \
            is not on a spherical surface with radius 1
    >>> projectCoordsOnSphere(coords)
    array([[ 0.        ,  0.        ,  1.        ],
           [ 0.        ,  1.        ,  0.        ],
           [ 0.57735027,  0.57735027,  0.57735027]])
    """
    t = _np.sqrt(1./ ((coords**2).sum(axis=-1)))
    out_coords = t[:,_np.newaxis]*coords
    return out_coords

#def meshCircle(d_samples=100):
#    """
#    Calculate a meshgrid containing the points of a circle
#    (center (0,0) and radius 1)
#    
#    Input:
#    ------
#    -- d_samples - number of points along the diameter
#
#    Output:
#    -------
#    -- coords - masked 2D array containing the coordinates in rows
#              - 1st column: x
#              - 2nd column: y
#       (everything otside a unit sphere is masked)
#    """
#    coords = []
#    x = y = _np.linspace(-1, 1, d_samples, endpoint=True)
#    X, Y = _np.meshgrid(x,y)
#    #X = X.ravel()
#    #Y = Y.ravel()
#    #remove points outside the circle
#    #inlier  = X**2 + Y**2 <= 1
#    #return _np.array([X[inlier], Y[inlier]]).T
#    X = _np.ma.masked_where(X**2 + Y**2 > 1, X)
#    Y = _np.ma.masked_where(X**2 + Y**2 > 1, Y)
#    return _np.ma.column_stack([_np.ma.ravel(X), _np.ma.ravel(Y)])

def projectCircleOnSphere(coords, projection = 'stereographic'):
    """
    Project 2d coordinates inside a unit circle on a sphere with
    center (0,0) and radius 1.
    
    Stereographic projection:
    Where the given circle is believed to be the on the plane
    crossing the equator (z=0) and the perspective point is the
    southpole (0,0,-1). Points inside the circle are projected
    onto the northern hemisphere, points outside the circle on
    the souther hemisphere.

    Orthographic projection:
    All points have to lie inside the circle and are projected
    on the northern hemisphere by keeping the xy coordinates and
    adding a z coordinate to result in a radius of 1.
    Poinst outside the circle result in nans.

    Input:
    ------
    -- coords - 2D array containing the coordinates in rows
              - 1st column: x
              - 2nd column: y
    -- projection - str - 'stereographic' (standard) or 'orthographic'
                          (explanations see above)
    Output:
    ------
    -- sphere_coords - masked 3D array containing the coordinates in
                       rows
                     - 1st column: x
                     - 2nd column: y
                     - 3rd column: z
     (all coordinates outside a unit sphere are masked)
    """
    if projection == 'stereographic':
        xy = 2 * coords / (1 + _np.sum(coords**2, 1))[:,_np.newaxis]
        z = (1 + _np.sum(-1 * coords**2, 1)) / (1 + _np.sum(coords**2, 1))
        return _np.ma.column_stack([xy,z])
    elif projection == 'orthographic':
        z = _np.sqrt(1 - _np.sum(coords**2,1))
        return _np.ma.column_stack([coords,z])
    else:
        raise ValueError('projection must be \"stereographic\" or' +
                '\"orthographic\"')

def projectSphereOnCircle(sphere_coords, projection="stereographic"):
    """
    Project coordinates that lie on the surface of a unit sphere
    with center (0,0,0) and radius 1 into a unit circle

    If projection == 'stereographic':
    The vertex of the sphere is (0,0,1). The projection is done from
    the "southpole" (0,0,-1) onto the plane with z=0.
    The northern hemisphere is hereby projected into the unit circle,
    the souther hemisphere outside the unit circle.

    If projection == 'orthographic':
    Only the northern hemisphere is mapped onto a plane through the
    equator by keeping the xy-coordinates. For coordinates on the 
    southern hemisphere nan is returned.

    Input:
    ------
    -- sphere_coords - 2D array containing the coordinates in rows
              - 1st column: x
              - 2nd column: y
              - 3rd column: z
    -- projection - str - 'stereographic' (standard) or 'orthographic'
                          (explanations see above)
    Output:
    -------
    -- coords - 2D array containing the coordinates in rows
                     - 1st column: x
                     - 2nd column: y
    """
    if projection == 'stereographic':
        return sphere_coords[:,:2] / (1+sphere_coords[:,2])[:,_np.newaxis]
    elif projection == 'orthographic':
        return _np.where((sphere_coords[:,2] < 0)[:,_np.newaxis] *
                _np.ones(sphere_coords.shape, bool),
                _np.nan, sphere_coords)[:,:2]
    else:
        raise ValueError('projection must be \"stereographic\" or' +
                '\"orthographic\"')

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
    # cosine of angle is the inner product divided by the product of
    # 2-norms
    cos_angle = (coords1.dot(coords2.T) /
            ((coords1**2).sum(1)[:,_np.newaxis] *
                (coords2**2).sum(1)[_np.newaxis]))
    N = _np.arange(1,n+1,1, dtype=float)
    if which == 'G':
        #evaluate Legendre polynomial
        c_g = ( 2*N + 1) / (N**2 + N)**m #coefficients for g
        #start with 1st (not 0st) polynomial
        c_g = _np.hstack([[0], c_g])
        result = _legval(cos_angle, c_g) / (4 * _np.pi) # this is G in the Perrin publication
    elif which == 'H':
        c_h = (-2*N - 1) / (N**2 + N)**(m-1) #coefficients for h
        #start with 1st (not 0st) polynomial
        c_h = _np.hstack([[0], c_h])
        result = -1 * _legval(cos_angle, c_h) / (4 * _np.pi)
    else:
        raise ValueError('which must be G or H')
    return result

def _sphereSpline(data, G_hh, G_hw=None, H_hw=None, smooth=0,
        type='Interpolation', buffersize=64):
    """
    Internal Function for computation of CSD
    and Interpolation

    -- data - 2dim with channels x datapoints
    -- G_hh - G between real electrodes (hh = have-have)
    -- G_hw - G between real and wanted electrodes (hw = have-wanted) -
              needed for interpolation
    -- H_hw - H between real and wanted electrodes (hw = have-wanted) -
              needed for CSD
    -- smooth - float - set to zero if no smoothing should be applied
    """
    # Prepare the relevant matrices
    # make it 2d if not already so
    data = data.reshape(data.shape[0],-1)
    p, n = data.shape
    # get number of batches
    num_batches = int(_np.ceil((data.nbytes / n ) /
        float(buffersize*1024**2)))
    delims = _np.linspace(0, n, num_batches+1, endpoint=True).astype(int)
    #add smoothing parameter to G
    G_hh[range(p), range(p)] = G_hh[range(p), range(p)] + smooth
    #change g to solve matrix system
    G_hh = _np.hstack([_np.ones([G_hh.shape[0],1], G_hh.dtype),
        G_hh])
    G_hh = _np.vstack([_np.ones([1,G_hh.shape[1]], G_hh.dtype),
        G_hh])
    G_hh[0,0] = 0
    out = []
    for i in xrange(num_batches):
        # check if G_hh is ill-conditioned
        if _cond(G_hh) >= 1./_np.finfo(_np.float64).eps:
            raise UserWarning('Interpolation problem is ill-conditioned.' +
                    '\n' + 'Consider increasing smoothing or order of' + 
                    ' Legendre polynomial.')
        #c0 is first row of C, the other cs are in the following rows of C,
        #each column is for one datapoint of n
        C = _solve(G_hh.astype(_np.float64), _np.vstack([_np.zeros(delims[i+1] - 
            delims[i], data.dtype), data[:,delims[i]:delims[i+1]]]).astype(
                _np.float64))
        ############################################################
        if type == 'Interpolation':
            out.append(C[0,:] + _np.ma.dot(G_hw.T, C[1:], strict=True))
        elif type == 'CSD':
            out.append(_np.ma.dot(H_hw.T, C[1:], strict=True))
        else: raise ValueError('Type must be Interpolation or CSD')
    result = _np.ma.hstack(out).astype(data.dtype)
    if _np.all(result.mask == False):
        return result.data
    else:
        return result

def _insideHull(points, vertices):
    """
    Find out if which points are inside a convex hull as defined by its
    vertices
    """
    codes = _np.ones(len(vertices)+1, _np.uint8) + 1
    vertices = _np.vstack([vertices, [0,0]])
    codes[0] = 1
    codes[-1] = 79
    path = _mpath.Path(vertices=vertices, codes=codes)
    return path.contains_points(points)

def potMap(RealCoords, data, diameter_samples=200, n=(7,7), m=4,
        smooth=0, projection='stereographic'):
    """
    Get a scalp map of potentials interpolated on a sphere.

    If projection == 'stereographic':
    The vertex of the sphere is (0,0,1). The projection is done from
    the "southpole" (0,0,-1) onto the plane with z=0.
    The northern hemisphere is hereby projected into the unit circle,
    the souther hemisphere outside the unit circle.

    If projection == 'orthographic':
    Only the northern hemisphere is mapped onto a plane through the
    equator by keeping the xy-coordinates. For coordinates on the 
    southern hemisphere nan is returned.
    
    Input:
    ------
    -- RealCoords - array of shape channels x 3 - (x,y,z) cartesian
                    coordinates of physical electrodes on a sphere with
                    center (0,0,0) and radius 1
    -- data - array of shape channels x datapoints - containing values
              at electrodes in same order as in 'RealCoords'
    -- diameter_samples - int - number of points along the diameter of
                          the scalp to interpolate
    -- n - tuple of ints - how many terms of the Legendre-Polynomial
           should be calculated, defaults to (7,7)
    -- m - int - order of the spherical spline interpolation, defaults
           to 4
    -- smooth - float - amount of smoothing, defaults to 0
    -- projection - str - 'stereographic' (standard) or 'orthographic'
                          (explanations see above)
    
    Output:
    -------
    -- X, Y, Z - grid containing the X and Y coordinates and the
                 interpolated values where everything outside the unit
                 circle is masked.
    """
    RealCoords = projectCoordsOnSphere(RealCoords)
    RealCoords_2D = projectSphereOnCircle(RealCoords, projection=projection)
    # get the convex hull of 2d points
    hull = _hull(RealCoords_2D[_np.all(_np.isfinite(RealCoords_2D),1)])
    # the final grid will be quadratic, so the extent of the grid is equal for x
    # and y
    grid_extent = _np.abs(RealCoords_2D[_np.all(_np.isfinite(RealCoords_2D),
        1)][hull.vertices]).max(None)
    # along the equator diameter_samples should be plotted,
    # if the grid extent is smaller or larger scale adequately
    grid_size = int(_np.ceil(diameter_samples * grid_extent))
    x = _np.linspace(-grid_extent, grid_extent, grid_size, endpoint=True)
    X, Y = _np.meshgrid(x,x)
    InterpCoords_2D = _np.column_stack([X.ravel(), Y.ravel()])
    #mask all points outside the hull of existing Coordinates
    mask = _insideHull(InterpCoords_2D, RealCoords_2D[_np.all(_np.isfinite(RealCoords_2D),1)][hull.vertices])
    InterpCoords_2D = _np.ma.masked_where(
            ~_np.column_stack([mask,mask]),
            InterpCoords_2D)
    InterpCoords_3D = projectCircleOnSphere(InterpCoords_2D)
    G_hh = _getGH(RealCoords, RealCoords, m=m, n=n[0], which='G')
    G_hw = _getGH(RealCoords, InterpCoords_3D, m=m, n=n[1], which='G')
    pot = _sphereSpline(data, G_hh=G_hh, G_hw=G_hw, smooth=smooth,
            type='Interpolation')
    return (InterpCoords_2D[:,0].reshape(grid_size, grid_size),
            InterpCoords_2D[:,1].reshape(grid_size, grid_size),
            pot.reshape(grid_size, grid_size))

def csdMap(RealCoords, data, diameter_samples=200, n=(7,20), m=4,
        smooth=1E-5, projection='stereographic'):
    """
    Get a scalp map of CSDs calculated from spherical splines.

    If projection == 'stereographic':
    The vertex of the sphere is (0,0,1). The projection is done from
    the "southpole" (0,0,-1) onto the plane with z=0.
    The northern hemisphere is hereby projected into the unit circle,
    the souther hemisphere outside the unit circle.

    If projection == 'orthographic':
    Only the northern hemisphere is mapped onto a plane through the
    equator by keeping the xy-coordinates. For coordinates on the 
    southern hemisphere nan is returned.
    
    Input:
    ------
    -- RealCoords - array of shape channels x 3 - (x,y,z) cartesian
                    coordinates physical electrodes on a sphere with
                    center (0,0,0) and radius 1
    -- data - array of shape channels x datapoints - containing values
              at electrodes in same order as in 'RealCoords'
    -- diameter_samples - int - number of points along the diameter of
                          the scalp to interpolate
    -- n - tuple of ints - how many terms of the Legendre-Polynomial
           should be calculated, defaults to (7,20)
    -- m - int - order of the spherical spline interpolation,
           defaults to 4
    -- smooth - float - amount of smoothing, defaults to 1E-5
    -- projection - str - 'stereographic' (standard) or 'orthographic'
                          (explanations see above)
    
    Output:
    -------
    -- X, Y, Z - grid containing the X and Y coordinates and the
                 interpolated values where everything outside the unit
                 circle is masked. the unit of z is data_unit / m**2
    """
    RealCoords = projectCoordsOnSphere(RealCoords)
    RealCoords_2D = projectSphereOnCircle(RealCoords, projection=projection)
    # get the convex hull of 2d points
    hull = _hull(RealCoords_2D[_np.all(_np.isfinite(RealCoords_2D),1)])
    # the final grid will be quadratic, so the extent of the grid is equal for x and y
    grid_extent = _np.abs(RealCoords_2D[_np.all(_np.isfinite(RealCoords_2D),1)][hull.vertices]).max(None)
    grid_size = diameter_samples * grid_extent # along the equator diameter_samples should be plotted,
                                               # if the grid extent is smaller or larger scale adequately
    x = _np.linspace(-grid_extent, grid_extent, grid_size, endpoint=True)
    X, Y = _np.meshgrid(x,x)
    InterpCoords_2D = _np.column_stack([X.ravel(), Y.ravel()])
    #mask all points outside the hull of existing Coordinates
    mask = _insideHull(InterpCoords_2D, RealCoords_2D[_np.all(_np.isfinite(RealCoords_2D),1)][hull.vertices])
    InterpCoords_2D = _np.ma.masked_where(
            ~_np.column_stack([mask,mask]),
            InterpCoords_2D)
    InterpCoords_3D = projectCircleOnSphere(InterpCoords_2D)
    G_hh = _getGH(RealCoords, RealCoords, m=m, n=n[0], which='G')
    H_hw = _getGH(RealCoords, InterpCoords_3D, m=m, n=n[1], which='H')
    pot = _sphereSpline(data, G_hh=G_hh, H_hw=H_hw, smooth=smooth,
            type='CSD')
    return (InterpCoords_2D[:,0].reshape(grid_size, grid_size),
            InterpCoords_2D[:,1].reshape(grid_size, grid_size),
            pot.reshape(grid_size, grid_size))

def calcCSD(Coords, data, n=(7,20), m=4, smooth=1E-5, buffersize=64):
    """
    Calculate current source densities at the same electrode posisionts
    as in the original data set
    
    Input:
    ------
    -- Coords - 2d array - channels x (x,y,z) cartesian coordinates of
                physical electrodes
    -- data - array containing values at electrodes in same order as in
              'Coords'
        -- 1st dim: channels
        -- 2nd dim: datapoints
    -- n - tuple of ints - How many terms of the Legendre-Polynomial
           should be calculated, defaults to (7, 20) - 7 for the first
           and 20 for the 2nd polynomial
    -- m - order of the spherical spline interpolation, defaults to 4
    -- smooth - float - amount of smoothing, defaults to 1E-5
    -- buffersize - float - determine the size of a buffer in MB to do
                    the calculations

    Output:
    -------
    -- pot - array containing CSD-values in data-unit / m**2

    Example:
    --------
    >>> Coords = getStandardCoordinates(['C1', 'C3'])
    >>> Coords = projectCoordsOnSphere(Coords) # be sure that the \
            points are on a sphere
    >>> data = _np.arange(6, dtype=float).reshape(2,-1)
    >>> calcCSD(Coords, data)
    array([[-3.5474569 , -3.5474569 , -3.5474569 ],
           [ 3.60884286,  3.60884286,  3.60884286]])
    """
    # be sure that these coords are on the surface of a unit sphere
    Coords = projectCoordsOnSphere(Coords)
    G_hh = _getGH(Coords, Coords, n=n[0], m=m, which='G')
    H_hw = _getGH(Coords, Coords, n=n[1], m=m, which='H')
    pot = _sphereSpline(data, G_hh = G_hh, H_hw = H_hw,
            buffersize=buffersize, type='CSD', smooth=smooth)
    return pot

def smoothSP(Coords, data, n=(7,7), m=4, smooth=1E-5, buffersize=64):
    """
    Calculate spatially smoothed potential values at the same electrode
    positions as in the original data set
    
    Input:
    ------
    -- Coords - 2d array - channels x (x,y,z) cartesian coordinates of
                physical electrodes
    -- data - array containing values at electrodes in same order as in
              'Coords'
        -- 1st dim: channels
        -- 2nd dim: datapoints
    -- n - tuple of ints - How many terms of the Legendre-Polynomial
           should be calculated, defaults to (7, 7) - 7 for the first
           and 7 for the 2nd polynomial
    -- m - order of the spherical spline interpolation, defaults to 4
    -- smooth - float - amount of smoothing, defaults to 1E-5
    -- buffersize - float - determine the size of a buffer in MB to do
                    the calculations

    Output:
    -------
    -- pot - array containing the spatially smoothed potentials
    
    Example:
    --------
    >>> Coords = getStandardCoordinates(['C1', 'C3'])
    >>> Coords = projectCoordsOnSphere(Coords) # be sure that the \
            points are on a sphere
    >>> data = _np.arange(6, dtype=float).reshape(2,-1)
    >>> smoothSP(Coords, data)
    array([[ 0.01220571,  1.01220571,  2.01220571],
           [ 2.98778429,  3.98778429,  4.98778429]])
    """
    # be sure that these coords are on the surface of a unit sphere
    Coords = projectCoordsOnSphere(Coords)
    G_hh = _getGH(Coords, Coords, n=n[0], m=m, which='G')
    G_hw = _getGH(Coords, Coords, n=n[1], m=m, which='G')
    pot = _sphereSpline(data, G_hh = G_hh, G_hw = G_hw,
            buffersize=buffersize, type='Interpolation', smooth=smooth)
    return pot
