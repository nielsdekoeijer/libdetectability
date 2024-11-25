from setuptools import setup, find_packages

setup(
    name="libdetectability",
    version="0.8.0",
    packages=find_packages(),
    install_requires=["pytest", "numpy", "scipy", "torch", "libsegmenter"],
    test_suite="test",
)
