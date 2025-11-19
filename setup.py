from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="minitorch-ki8yk8",
    version="1.0.2",
    author="KI8YK8",
    description="Implementing minimal Pytorch for Learning",
		long_description=long_description,
		long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
                "matplotlib>=3.9.0",
                "numpy>=2.0.0",
								"pyqt6"
    ],
    python_requires=">=3.9",
)
