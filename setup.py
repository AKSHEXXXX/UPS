from setuptools import setup, find_packages

setup(
    name="coldchain-gym",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "ppo": ["stable-baselines3>=2.0.0"],
        "train": ["stable-baselines3>=2.0.0"],
        "dev": ["pytest>=7.0.0"],
    },
)
