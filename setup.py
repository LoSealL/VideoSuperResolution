from setuptools import find_packages
from setuptools import setup

VERSION = '0.2.14'

REQUIRED_PACKAGES = [
    'numpy',
    'Image',
    # 'tensorflow',
]

if __name__ == '__main__':
    setup(
        name='VSR',
        version=VERSION,
        description='Video Super-Resolution Framework',
        url='https://github.com/LoSealL/VideoSuperResolution',
        packages=find_packages(),
        install_requires=REQUIRED_PACKAGES,
        license='MIT',
        maintainer='Wenyi Tang',
        maintainer_email='wenyitang@outlook.com'
    )
