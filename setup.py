import os

from setuptools import setup, find_packages

NAME = "temporal_logic"
DESCRIPTION = ""
URL = ""
EMAIL = "anandbal@usc.edu"
AUTHOR = "Anand Balakrishnan"

REQUIRES_PYTHON = '>=3.3'
VERSION = '0.1.0'
REQUIRED_PKGS = [
    "numpy",
    "scipy",
    "cython",
    "torch",
    "sympy",
]

EXTRAS = {}

EXTENSIONS = []

HERE = os.path.abspath(os.path.dirname(__file__))

ABOUT = dict()
ABOUT['__version__'] = VERSION

setup(
    name=NAME,
    version=ABOUT['__version__'],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests', 'scripts', 'experiments',)),

    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],

    # ext_modules=cythonize(EXTENSIONS),
)
