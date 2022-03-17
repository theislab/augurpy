augurpy
===========================

|PyPI| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/augurpy.svg
   :target: https://pypi.org/project/augurpy/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/augurpy
   :target: https://pypi.org/project/augurpy
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/theislab/augurpy
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/augurpy/latest.svg?label=Read%20the%20Docs
   :target: https://augurpy.readthedocs.io/
   :alt: Read the documentation at https://augurpy.readthedocs.io/
.. |Build| image:: https://github.com/theislab/augurpy/workflows/Build%20augurpy%20Package/badge.svg
   :target: https://github.com/theislab/augurpy/actions?workflow=Package
   :alt: Build Package Status
.. |Tests| image:: https://github.com/theislab/augurpy/workflows/Run%20augurpy%20Tests/badge.svg
   :target: https://github.com/theislab/augurpy/actions?workflow=Tests
   :alt: Run Tests Status
.. |Codecov| image:: https://codecov.io/gh/theislab/augurpy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/theislab/augurpy
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black


the python implementation of `Augur R package <https://github.com/neurorestore/Augur>`_ Skinnider, M.A., Squair, J.W., Kathe, C. et al. `Cell type prioritization in single-cell data <https://doi.org/10.1038/s41587-020-0605-1>`_. Nat Biotechnol 39, 30–34 (2021).

Augurpy aims to rank or prioritize cell types according to the their response to experimental perturbations given high dimensional single-cell sequencing data. The basic idea is that in the space of molecular measurements cells reacting heavily to induced perturbations are more easily seperated into perturbed and unperturbed than cell types with little or no response. This seperability is quantified by measuring how well experimental labels (eg. treatment and control) can be predicted within each cell type. Augurpy trains a machine learning model predicting experimental labels for each cell type in multiple cross validation runs and then prioritizes cell type response according to metric scores measuring the accuracy of the model. For categorical data the area under the curve is the default metric and for numerical data the concordance correlation coefficient is used as a proxy for how accurate the model is which in turn approximates perturbation response.


Installation
------------

You can install *augurpy* via pip_ from PyPI_:

.. code:: console

   $ pip install augurpy


Usage
-----

Please see the `Command-line Reference <Usage_>`_ for details.


Credits
-------

This package was created with cookietemple_ using Cookiecutter_ based on Hypermodern_Python_Cookiecutter_.

.. _cookietemple: https://cookietemple.com
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _PyPI: https://pypi.org/
.. _Hypermodern_Python_Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _pip: https://pip.pypa.io/
.. _Usage: https://augurpy.readthedocs.io/en/latest/usage.html
