========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |codecov|
    * - package
      - |version| |wheel| |supported-versions| |supported-implementations| |commits-since|
.. |docs| image:: https://readthedocs.org/projects/inter-active-learning/badge/?style=flat
    :target: https://readthedocs.org/projects/inter-active-learning/
    :alt: Documentation Status

.. |codecov| image:: https://codecov.io/gh/milosz-l/inter-active-learning/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/milosz-l/inter-active-learning

.. |version| image:: https://img.shields.io/pypi/v/inter-active-learning.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/inter-active-learning

.. |wheel| image:: https://img.shields.io/pypi/wheel/inter-active-learning.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/inter-active-learning

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/inter-active-learning.svg
    :alt: Supported versions
    :target: https://pypi.org/project/inter-active-learning

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/inter-active-learning.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/inter-active-learning

.. |commits-since| image:: https://img.shields.io/github/commits-since/milosz-l/inter-active-learning/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/milosz-l/inter-active-learning/compare/v0.0.0...main



.. end-badges

Interactive Active Learning Python package: Label less and learn more with any algorithm by strategically labeling only
key samples.

* Free software: MIT license

Installation
============

::

    pip install inter-active-learning

You can also install the in-development version with::

    pip install https://github.com/milosz-l/inter-active-learning/archive/main.zip


Documentation
=============


https://inter-active-learning.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
