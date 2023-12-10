# from jitcsim.__init__ import __version__ as v
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='jitcsim',
<<<<<<< HEAD
    version="0.3",
    author="Abolfazl Ziaeemehr",
=======
    version='0.3.1',
    author="...",
>>>>>>> ed5ee808b6480ba3292b9b9a1b5aec04e6f47abd
    author_email="a.ziaeemehr@gmail.com",
    description="Simulation of networks with Ordinary/Delay/Stochastic differential equations (just-in-time compilation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ziaeemehr/JITCSIM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    # python_requires='>=3.8',
    # package_data={'sbi_nmms': ['DampOscillator.so']},
    # install_requires=requirements,
    # include_package_data=True,
)
