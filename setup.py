import os
from setuptools import setup


def read(filename):
    with open(filename) as f:
        return f.read()


setup(
    name="fill_holes",
    version="0.1",
    author="Chris Lucas",
    author_email="chris.lucas@geodan.nl",
    description=(
        "A script to fill holes in a point cloud with synthetic points."),
    license="MIT",
    keywords="fill holes point cloud synthetic data",
    packages=['fill_holes'],
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'numba'
    ],
    zip_safe=False
)