import os
from setuptools import setup
from setuptools import find_packages

setup(
    name="dmc_to_gym",
    version="1.0.0",
    author="Rory Young",
    description=("a gym like wrapper for dm_control"),
    license="",
    keywords="gym dm_control openai deepmind",
    packages=find_packages(),
    install_requires=[
        "gym",
        "dm_env",
        "numpy",
    ],
)
