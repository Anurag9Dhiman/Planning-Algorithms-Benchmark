from setuptools import setup, find_packages

setup(
    name="planning-benchmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "tqdm>=4.65",
    ],
    python_requires=">=3.9",
)
