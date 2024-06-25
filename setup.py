import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name = 'richvalues',
    version = '4.1.1',
    license = 'BSD-3-Clause',
    author = 'Andrés Megías Toledano',
    description = 'Python library for working with uncertainties and upper/lower limits',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages = setuptools.find_packages('.'),
    url = 'https://github.com/andresmegias/richvalues/',
    install_requires = ['numpy', 'pandas', 'scipy', 'matplotlib'],
    classifiers = ['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent'],
    project_urls = {"Documentation": "https://github.com/andresmegias/richvalues/blob/main/userguide.pdf"}
)
