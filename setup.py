from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name = 'richvalues',
    version = '1.0.5',
    license = 'BSD-3-Clause',
    author = 'Andrés Megías Toledano',
    description = 'Python library for dealing with uncertainties and upper/lower limits',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages = find_packages('src'),
    package_dir = {'': 'src'},
    url = 'https://github.com/andresmegias/richvalues',
    install_requires = ['numpy', 'pandas']
)
