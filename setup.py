#!/usr/bin/env python

from setuptools import setup

setup(name='meet',
      version='20140408',
      description='Modular EEg Toolkit (MEET)',
      author='Gunnar Waterstraat',
      author_email='gunnar.waterstraat@gmx.de',
      url='https://github.com/t00tsie/meet',
      packages=['meet'],
      install_requires=['numpy', 'scipy', 'matplotlib']
     )
