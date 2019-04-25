from setuptools import setup

__version__ = "0.9.0"


setup(
    name="pulp_lparray",
    python_requires=">=3.7.0",
    install_requires=["pulp"],
    tests_require=["pytest"],
    packages=["pulp_lparray"],
    version=__version__,
)
