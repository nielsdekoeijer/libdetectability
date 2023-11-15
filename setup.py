from setuptools import setup, find_packages

setup(
    name='libdetectability',
    version='0.2.2',
    packages=find_packages(),
    install_requires=[
        "pytest", "numpy", "scipy", "torch"
    ],
    test_suite='test',
)
