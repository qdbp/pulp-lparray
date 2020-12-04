from setuptools import setup

__version__ = "1.0.20201204002401"


setup(
    name="pulp_lparray",
    python_requires=">=3.9.0",
    install_requires=["pulp", "numpy"],
    tests_require=["pytest"],
    packages=["pulp_lparray"],
    version=__version__,
)
