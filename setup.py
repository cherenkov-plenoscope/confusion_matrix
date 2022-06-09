import setuptools

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="confusion_matrix_sebastian-achim-mueller",
    version="0.0.3",
    description="Create confusion-matrices",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/confusion_matrix",
    project_urls={
        "Bug Tracker": "https://github.com/cherenkov-plenoscope/confusion_matrix/issues",
    },
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=["confusion_matrix",],
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
    ],
)
