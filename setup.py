#!/usr/bin/env python

from setuptools import setup

setup(name='meet',
      version='20150102',
      description='Modular EEg Toolkit (MEET)',
      author='Gunnar Waterstraat',
      author_email='gunnar.waterstraat@gmx.de',
      url='https://github.com/t00tsie/meet',
      packages=['meet'],
      package_data={'meet': ['plotting_1005.txt', 'test_data/elecnames.txt', 'test_data/sample.dat']},
      install_requires=['numpy', 'scipy', 'matplotlib']
     )
