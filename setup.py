from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="lasertools_rffthelper",
    version="0.0.1",
    description="A module to handle RFFT computations",
    license="GPLv3",
    long_description=long_description,
    author="brittonm",
    author_email="68578865+brittonm@users.noreply.github.com",
    url="https://github.com/lasertools/lasertools_rffthelper",
    download_url = 'https://github.com/lasertools/lasertools_rffthelper/archive/refs/tags/v_0_0_1.tar.gz',
    packages=["lasertools_rffthelper"],
    install_requires=[
        "numpy",
        "scipy",
    ],
  classifiers=[
    'Programming Language :: Python :: 3.9',
  ],
)
