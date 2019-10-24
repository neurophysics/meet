'''
This is the Modular EEg Toolkit (MEET) for Python 2.

************************************************************************
***Disclosure:                                                       ***
***-----------                                                       ***
***This software comes as it is - there might be errors at runtime   ***
***and results might be wrong although the code was tested and did   ***
***work as expected. Since resuluts might be wrong you must          ***
***absolutely not use this software for a medical purpuse - decisions***
***converning diagnosis, treatment or prophylaxis of any medical     ***
***condition mustn't rely on this software.                          ***
************************************************************************

License:
--------
Copyright (c) 2015 Gunnar Waterstraat

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Author & Contact
----------------
Written by Gunnar Waterstraat
email: gunnar[dot]waterstraat[at]charite.de
'''

import numpy as _np
from os import path as _path
import scipy.signal as _signal
import scipy.linalg as _linalg
import scipy.interpolate as _sci

_packdir = _path.dirname(_path.abspath(__file__))

__all__ = ['basic', 'eeg_viewer', 'elm', 'iir', 'spatfilt', 'sphere', 'tf']

#import all basic functions into base namespace
from .basic import *

#import submodules automatically
from . import eeg_viewer
from . import elm
from . import iir
from . import spatfilt
from . import sphere
from . import tf
