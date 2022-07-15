Blocks
======

.. doxygentypedef:: eqs_block_t

The following functions operate on :c:type:`eqs_block_t`:

- :c:func:`eqs_block`: create new blocks
- :c:func:`eqs_block_free`: free allocated blocks
- :c:func:`eqs_block_labels`: get one of the :c:struct:`eqs_labels_t` associated with this block
- :c:func:`eqs_block_data`: get one of the :c:struct:`eqs_array_t` associated with this block
- :c:func:`eqs_block_add_gradient`: add gradient data to this block
- :c:func:`eqs_block_gradients_list`: get the list of gradients in this block

---------------------------------------------------------------------

.. doxygenfunction:: eqs_block

.. doxygenfunction:: eqs_block_free

.. doxygenfunction:: eqs_block_labels

.. doxygenfunction:: eqs_block_data

.. doxygenfunction:: eqs_block_add_gradient

.. doxygenfunction:: eqs_block_gradients_list
