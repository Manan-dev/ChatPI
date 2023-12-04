from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()
    
setup(
    name="chatpi",
    version="0.1",
    url="https://github.com/Manan-dev/ChatPI",
    packages=find_packages(include=["chatpi"]),
    install_requires=required,
)