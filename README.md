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
## Links
- Library project on PyPI: https://pypi.org/project/richvalues/.
- Library project on Anaconda website: https://anaconda.org/richvalues/richvalues/.

## Quick tutorial
Here you can see how to use the library within a Python script or terminal. For more details, please check the [user guide](https://github.com/andresmegias/richvalues/blob/main/userguide.pdf).

First of all, we import the library.
~~~
import richvalues as rv
~~~
Let's create two rich values.
~~~
x = rv.rval('5.2 +/- 0.4')
y = rv.rval('3.1 -0.4+0.5')
~~~
If you want to specify a domain, you can write it between brackets, for example: `rv.rval('5.2 +/-0.4 [0,inf]')`. You can access to the properties of rich values; for example: `x.main` would return the main value, `5.2`, and `x.unc` would return the inferior and superior uncertainties, `[0.4, 0.4]`. You can also obtain derived properties, like the relative uncertainty, `x.rel_unc`.

Now, we can use mathematical operators to perform calculations with these rich values. For example, `x + y` would yield `8.3 +/- 0.6`. Alternatively, you can use `function`, that allows to apply more complicated functions.
~~~
rv.function('{}+{}', args=[x,y])
~~~
You just have to write the expression that you want to apply, using empty curly brackets instead of the inputs, which have to be specified in the correct order. The function expression can include other functions, for example, if you imported the NumPy library as `np`, you could write: `rv.function('np.sin({}/{})', [x,y])`.

Now, let's see how to create rich arrays (based on NumPy arrays).
~~~
u = rv.rarray(['1.2 +/- 0.4', '5.8 +/-0.9'])
v = rv.rarray(['8 +/- 3', '< 21'])
~~~
The domain can be specified with de `domain` argument. As with individual rich values, you can access to different properties; for example, `u.mains` would return the main values, `[8., 21.]`. You can use arithmetic operators as well to perform calculations; for example, `u*v` would yield `[9-4+5, < 150]`. Alternatively, you can use `array_function`.
~~~
rv.array_function('{}*{}, [u,v], elementwise=True)
~~~
Lastly, let's see how to create a rich dataframe (based on Pandas dataframes). The easiest way is to convert a dataframe with text strings representing rich values using `rich_dataframe`, but you can also convert dictionaries.
~~~
dic = {'a': ['2.1+/-0.3','5'], 'b': ['3.4+/-0.4','<6'], 'c': ['<4','8+/-1']}
rdf = rv.rich_dataframe(dic, domains=[0,np.inf])
~~~
You can access to different properties of the values of the rich dataframe; for example, `rdf.mains` would return a regular dataframe containing the main values of the elements of `rdf`. Arithmetic operators can be used with rich dataframes, although for more complicated functions you can use `create_column` and `create_row`.
~~~
rdf['d'] = rdf.create_column('np.tan({}/{})', ['a','b'])
~~~
Note that in this case you have to specify the names of the columns involved in the calculation.

That would be it for this quick tutorial. If you want to learn more, you can read the [user guide](https://github.com/andresmegias/richvalues/blob/main/userguide.pdf) and also check and run the [example scripts](https://github.com/andresmegias/richvalues/tree/main/examples) (`ratio.py` and `linearfit.py`).
