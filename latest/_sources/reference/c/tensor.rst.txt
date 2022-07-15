Tensor Maps
===========

.. doxygentypedef:: eqs_tensormap_t

The following functions operate on :c:type:`eqs_tensormap_t`:

- :c:func:`eqs_tensormap`: create new tensor map
- :c:func:`eqs_tensormap_free`: free allocated tensor maps
- :c:func:`eqs_tensormap_keys`: get the keys defined in a tensor map as :c:struct:`eqs_labels_t`
- :c:func:`eqs_tensormap_block_by_id`: get a :c:struct:`eqs_block_t` in a tensor map from its index
- :c:func:`eqs_tensormap_blocks_matching`: get a list of block indexes matching a selection
- :c:func:`eqs_tensormap_keys_to_samples`: move entries from keys to sample labels
- :c:func:`eqs_tensormap_keys_to_properties`: move entries from keys to properties labels
- :c:func:`eqs_tensormap_components_to_properties`: move entries from component labels to properties labels


---------------------------------------------------------------------

.. doxygenfunction:: eqs_tensormap

.. doxygenfunction:: eqs_tensormap_free

.. doxygenfunction:: eqs_tensormap_keys

.. doxygenfunction:: eqs_tensormap_block_by_id

.. doxygenfunction:: eqs_tensormap_blocks_matching

.. doxygenfunction:: eqs_tensormap_keys_to_samples

.. doxygenfunction:: eqs_tensormap_keys_to_properties

.. doxygenfunction:: eqs_tensormap_components_to_properties
