from setuptools import setup, find_packages

setup(name="pyrmm",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires=">=3",
    )