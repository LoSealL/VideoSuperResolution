from setuptools import find_packages
from setuptools import setup

version = '0.1.6'

REQUIRED_PACKAGES = [
    'numpy',
    'tensorflow-gpu',
]

setup(
    name='VSR',
    version=version,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    author='Wenyi Tang'
)
