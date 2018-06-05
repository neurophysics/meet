#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='meet',
      version='20180605',
      description='Modular EEg Toolkit (MEET)',
      author='Gunnar Waterstraat',
      author_email='gunnar.waterstraat@charite.de',
      url='https://github.com/neurophysics/meet',
      packages=['meet'],
      package_data={'meet': ['plotting_1005.txt', 'test_data/elecnames.txt', 'test_data/sample.dat']},
      install_requires=["numpy","scipy","matplotlib"],
      setup_requires=["numpy"],
     )
