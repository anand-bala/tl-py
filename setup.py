from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "sympy ~= 1.3",
    "pandas ~= 0.24.1",
    "numpy ~= 1.16",
    "scipy ~= 1.2",
    "future",
]

setup(
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=2.6, !=3.0.*, !=3.1.*, !=3.2.*, <4"
)
