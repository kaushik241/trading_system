from setuptools import setup, find_packages

setup(
    name="trading_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dateutil",
        "pandas",
        "numpy",
        "matplotlib",
    ],
)
