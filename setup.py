from setuptools import setup

__version__ = "0.10.1"


setup(
    name="pulp_lparray",
    python_requires=">=3.8.0",
    install_requires=["pulp", "numpy"],
    tests_require=["pytest"],
    packages=["pulp_lparray"],
    version=__version__,
)
