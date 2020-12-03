import setuptools
import os

path = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(path, "../README.md")
with open(readme_path, "r") as fh:
    long_description = fh.read()

print("the packages are:")
print(setuptools.find_packages())

setuptools.setup(
    name="musweeper",
    version="0.0.1",
    author="Sigurd Schoeler and Brage Arnkværn",
    description="MuZero plays MineSweeper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zikoat/musweeper",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)