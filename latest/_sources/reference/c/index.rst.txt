.. _c-api-reference:

C API reference
===============

``equistore`` offers a C API that can be called from any language able to call
C functions (in particular, this includes Python, Fortran with ``iso_c_env``,
C++, and most languages used nowadays). Convenient wrappers of the C API are
also provided for :ref:`Python <python-api-reference>` users.

The C API is implemented in Rust. You can use these functions in your own code
by :ref:`installing the corresponding shared library and header
<install-c-lib>`, and then including ``equistore.h`` and linking with
``-lequistore``. Alternatively, we provide a cmake package config file, allowing
you to do use ``equistore`` like this (after installation):

.. code-block:: cmake

    find_package(equistore)

    # add executables/libraries
    add_executable(MyExecutable my_sources.c)
    add_library(MyLibrary my_sources.c)

    # Links equistore with MyExecutable, this makes the header accessible
    target_link_libraries(MyExecutable equistore)

The functions and types provided in ``equistore.h`` can be grouped in five
main groups:

.. toctree::
    :maxdepth: 1

    tensor
    labels
    blocks
    data
    misc
