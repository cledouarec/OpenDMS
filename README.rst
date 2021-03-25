*******
OpenDMS
*******

Overview
--------

OpenDMS is a project to implement a simple driver monitoring system.
This project is not intended to be use as a production system but to implement
basic concepts.

Installation
------------

It is recommended to use a virtual environment ::

    python -m venv venv

To use (with caution), simply do ::

    pip install .

For the developers, it is useful to install extra tools like :

* pre-commit : https://pre-commit.com
* pytest : http://docs.pytest.org

These tools can be installed with the following command ::

    pip install .[dev]

The Git hooks can be installed with ::

    pre-commit install

The hooks can be run manually at any time ::

    pre-commit run --all-file

Usage
-----

In progress.
