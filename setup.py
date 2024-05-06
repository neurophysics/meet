#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='meet',
      version='2024.05.06',
      description='Modular EEg Toolkit (MEET)',
      author='Gunnar Waterstraat',
      author_email='gunnar.waterstraat@charite.de',
      url='https://github.com/neurophysics/meet',
      packages=['meet'],
      package_data={'meet': [
          'plotting_1005.txt', 'test_data/elecnames.txt',
          'test_data/sample.dat']},
      install_requires=["numpy","scipy","matplotlib"],
      setup_requires=["numpy"],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Framework :: IPython',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Environment :: Web Environment',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Intended Audience :: System Administrators',
          'License :: OSI Approved :: Python Software Foundation License',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          ]
     )
