# RichValues

Python 3 library for dealing with numeric values with uncertainties, upper/lower limits and finite intervals, which may be called _rich values_.

With it, one can import rich values written in plain text documents in an easily readable format, operate with them propagating the uncertainties automatically, and export them in the same formatting style as the import. It also allows to easily plot rich values and to make fits to any function taking into account the uncertainties and upper/lower limits or finite intervals. Moreover, correlations between variables (that is, variables which are not independent) are taken into account when performing calculations with rich values.

The libraries NumPy, Pandas, SciPy and Matplotlib are required.

You can read the user guide ([`userguide.pdf`](https://github.com/andresmegias/richvalues/blob/main/userguide.pdf)) to learn how to use it. You can also test the library in the example scripts inside the `examples` folder.

## Installation

To install the library, you can use the Python Package Installer to download it from the Python Package Index (PyPI). To do so, run from the terminal the following command:
~~~
pip3 install richvalues
~~~
Alternatively, you can use the Conda package installer:
~~~
conda install richvalues -c richvalues
~~~

## Tutorial
You can check this [quick tutorial](https://github.com/andresmegias/richvalues/blob/main/tutorial.md) to learn the basics of this library.

## Links
- Library project on PyPI: https://pypi.org/project/richvalues/.
- Library project on Anaconda website: https://anaconda.org/richvalues/richvalues/.
