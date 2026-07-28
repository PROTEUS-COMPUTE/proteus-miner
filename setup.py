from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="proteus-miner",
    version="0.1.0",
    description="PROTEUS miner and validator neurons - Nvidia GPU only",
    url="https://github.com/PROTEUS-COMPUTE/proteus-miner",
    author="Proteus Compute",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
