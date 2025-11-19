from setuptools import setup, find_packages

setup(
    name="minitorch-ki8yk8",
    version="1.0.0",
    author="KI8YK8",
    description="Implementing minimal Pytorch for Learning",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
                "matplotlib>=3.10.7",
                "numpy>=2.2.6",
    ],
    python_requires=">=3.9",
)
