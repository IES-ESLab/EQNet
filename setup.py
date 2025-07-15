from setuptools import setup, find_packages

setup(
    name="EQNet",
    version="0.1.7",
    long_description="EQNet: Neural Network Models for Earthquakes",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["torch", "torchvision", "h5py", "matplotlib", "pandas"],
)
