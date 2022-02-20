"""
This module configures the package for distribution and installation.
"""

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="pynewton",
    version="0.0.0",
    packages=find_packages(),
    install_requires=["pygame", "pillow"],
    ext_modules=cythonize("pynewton/cynewton.pyx", language="c++"),
    entry_points={
        "console_scripts": [
            "pynewton = pynewton.__main__:main",
        ]
    },
)
