from setuptools import find_packages
from setuptools import setup

VERSION = '0.1.6'

REQUIRED_PACKAGES = [
    'numpy',
    'tensorflow-gpu',
]

setup(
    name='VSR',
    version=VERSION,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    author='Wenyi Tang'
)
