# RichValues

Python 3 library for dealing with numeric values with uncertainties and upper/lower limits, which we call _rich values_. With it, one can import rich values written in plain text documents in an easily readable format, operate with them propagating the uncertainties automatically, and export them in the same formatting style as the import.

This library require the modules `copy`, `math`, and `itertools`, from the Python’s standard library, and also the libraries NumPy and Pandas.

You can read the user guide (`richvalues-userguide.pdf`) to learn how to use it. You can also test the library in the example script inside the `example` folder.

## Installation

To install the library, you can use the Python Package Installer to download it from the Python Package Index (PyPI). To do so, run from the terminal the following command:
~~~
pip3 install richvalues
~~~
Alternatively, you can use the Conda package installer:
~~~
conda install richvalues -c richvalues
~~~
Links:
* Library project on PyPI: https://pypi.org/project/richvalues/
* Library project on Anaconda website: https://anaconda.org/richvalues/richvalues 
