import ctypes
from typing import List, Union

import numpy as np

from ._c_api import c_uintptr_t, eqs_block_t, eqs_labels_t
from ._c_lib import _get_library
from .block import TensorBlock
from .labels import Labels, _is_namedtuple, _print_labels
from .status import _check_pointer


class TensorMap:
    """
    A TensorMap is the main user-facing class of this library, and can store
    any kind of data used in atomistic machine learning.

    A tensor map contains a list of :py:class:`TensorBlock`, each one associated with
    a key. Users can access the blocks either one by one with the
    :py:func:`TensorMap.block` function, or by iterating over the tensor map
    itself:

    .. code-block:: python

        for key, block in tensor:
            ...

    A tensor map provides functions to move some of these keys to the
    samples or properties labels of the blocks, moving from a sparse
    representation of the data to a dense one.
    """

    def __init__(self, keys: Labels, blocks: List[TensorBlock]):
        """
        :param keys: keys associated with each block
        :param blocks: set of blocks containing the actual data
        """
        self._lib = _get_library()

        blocks_array_t = ctypes.POINTER(eqs_block_t) * len(blocks)
        blocks_array = blocks_array_t(*[block._ptr for block in blocks])

        # all blocks are moved into the tensor map, assign NULL to `block._ptr`
        # to prevent accessing the blocks from Python/double free
        for block in blocks:
            if not block._owning:
                raise ValueError(
                    "can not use blocks from another tensor map in a new one, "
                    "use TensorBlock.copy() to make a copy of each block first"
                )

            block._ptr = ctypes.POINTER(eqs_block_t)()

        # keep a reference to the blocks in the tensor map in case they contain
        # a Python-allocated array that we need to keep alive
        self._blocks = blocks

        self._ptr = self._lib.eqs_tensormap(
            keys._as_eqs_labels_t(), blocks_array, len(blocks)
        )
        _check_pointer(self._ptr)

    @staticmethod
    def _from_ptr(ptr):
        """Create a tensor map from a pointer owning its data"""
        _check_pointer(ptr)
        obj = TensorMap.__new__(TensorMap)
        obj._lib = _get_library()
        obj._ptr = ptr
        obj._blocks = []
        return obj

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_ptr"):
            self._lib.eqs_tensormap_free(self._ptr)

    def __iter__(self):
        keys = self.keys
        for i, keys in enumerate(keys):
            yield keys, self._get_block_by_id(i)

    def __len__(self):
        return len(self.keys)

    def __repr__(self) -> str:
        result = f"TensorMap with {len(self)} blocks\n"
        result += _print_labels(self.keys, header="keys")
        return result

    def __str__(self) -> str:
        result = f"TensorMap with {len(self)} blocks\n"
        result += _print_labels(
            self.keys,
            header="keys",
            print_limit=len(self) + 1,
        )
        return result

    @property
    def keys(self) -> Labels:
        """The set of keys labeling the blocks in this tensor map."""
        result = eqs_labels_t()
        self._lib.eqs_tensormap_keys(self._ptr, result)
        return Labels._from_eqs_labels_t(result, parent=self)

    def block(self, *args, **kwargs) -> TensorBlock:
        """
        Get the block in this tensor map matching the selection made with
        positional and keyword arguments.

        There are a couple of different ways to call this function:

        .. code-block:: python

            # with a numeric index, this gives a block by its position
            block = tensor.block(3)
            # this block corresponds to tensor.keys[3]

            # with a key
            block = tensor.block(tensor.keys[3])

            # with keyword arguments selecting the block
            block = tensor.block(key=-3, symmetric=4)
            # this assumes `tensor.keys.names == ("key", "symmetric")`

            # with Labels containing a single entry
            labels = Labels(
                names=["key", "symmetric"],
                values=np.array([[-3, 4]], dtype=np.int32)
            )
            block = tensor.block(labels)
        """

        if args and isinstance(args[0], int):
            return self._get_block_by_id(args[0])

        matching, selection = self.blocks_matching(
            *args, **kwargs, __return_selection=True
        )

        if len(matching) == 0:
            selection = next(selection.as_namedtuples())
            raise ValueError(
                f"Couldn't find any block matching the selection {selection.as_dict()}"
            )
        elif len(matching) > 1:
            selection = next(selection.as_namedtuples())
            raise ValueError(
                f"more than one block matched {selection.as_dict()}, use "
                "`TensorMap.blocks` if you want to get all of them"
            )
        else:
            return self._get_block_by_id(matching[0])

    def blocks(self, *args, **kwargs) -> List[TensorBlock]:
        """
        Get blocks in this tensor map, matching the selection made with
        positional and keyword arguments. Returns a list of all blocks matching
        the criteria.

        There are a couple of different ways to call this function:

        .. code-block:: python

            # with a numeric index, this gives a block by its position
            blocks = tensor.blocks(3)
            # this block corresponds to tensor.keys[3]

            # with a key
            blocks = tensor.blocks(tensor.keys[3])

            # with keyword arguments selecting the blocks
            blocks = tensor.blocks(key=-3, symmetric=4)
            # this assumes `tensor.keys.names == ("key", "symmetric")`

            # with Labels containing a single entry
            labels = Labels(
                names=["key"],
                values=np.array([[-3]], dtype=np.int32)
            )
            blocks = tensor.blocks(labels)
        """

        if args and isinstance(args[0], int):
            return [self._get_block_by_id(args[0])]

        matching, selection = self.blocks_matching(
            *args, **kwargs, __return_selection=True
        )

        if len(matching) == 0:
            selection = next(selection.as_namedtuples())
            raise ValueError(
                f"Couldn't find any block matching the selection {selection.as_dict()}"
            )
        else:
            return [self._get_block_by_id(i) for i in matching]

    def blocks_matching(self, *args, **kwargs) -> List[int]:
        """
        Get a (possibly empty) list of block indexes matching the selection made
        with positional and keyword arguments. This function can be called with
        different kinds of argument, similarly to :py:func:`TensorMap.block`.
        """
        return_selection = kwargs.pop("__return_selection", False)

        selection = None
        if args:
            if len(args) > 1:
                raise ValueError("only one non-keyword argument is supported")

            arg = args[0]
            if isinstance(arg, Labels):
                selection = arg
            elif isinstance(arg, np.void):
                # single entry from an Labels array
                return self.blocks_matching(
                    **{name: arg[i] for i, name in enumerate(arg.dtype.names)},
                    __return_selection=return_selection,
                )
            elif _is_namedtuple(arg):
                return self.blocks_matching(
                    **arg.as_dict(),
                    __return_selection=return_selection,
                )
            else:
                raise ValueError(
                    f"got unexpected object in `TensorMap.blocks_matching`: {type(arg)}"
                )

        if selection is None:
            selection = Labels(
                kwargs.keys(),
                np.array(list(kwargs.values()), dtype=np.int32).reshape(1, -1),
            )

        block_indexes = ctypes.ARRAY(c_uintptr_t, len(self.keys))()
        count = c_uintptr_t(block_indexes._length_)

        self._lib.eqs_tensormap_blocks_matching(
            self._ptr,
            block_indexes,
            count,
            selection._as_eqs_labels_t(),
        )

        result = []
        for i in range(count.value):
            result.append(int(block_indexes[i]))

        if return_selection:
            return result, selection
        else:
            return result

    def _get_block_by_id(self, id) -> TensorBlock:
        block = ctypes.POINTER(eqs_block_t)()
        self._lib.eqs_tensormap_block_by_id(self._ptr, block, id)
        return TensorBlock._from_ptr(block, parent=self, owning=False)

    def keys_to_properties(
        self,
        keys_to_move: Union[str, List[str], Labels],
        *,
        sort_samples=True,
    ):
        """
        Merge blocks with the same value for selected keys variables along the
        property axis.

        The variables (names) of ``keys_to_move`` will be moved from the keys to
        the property labels, and blocks with the same remaining keys variables
        will be merged together along the property axis.

        If ``keys_to_move`` does not contains any entries (i.e.
        ``keys_to_move.shape[0] == 0``), then the new property labels will
        contain entries corresponding to the merged blocks only. For example,
        merging a block with key ``a=0`` and properties ``p=1, 2`` with a block
        with key ``a=2`` and properties ``p=1, 3`` will produce a block with
        properties ``a, p = (0, 1), (0, 2), (2, 1), (2, 3)``.

        If ``keys_to_move`` contains entries, then the property labels must be
        the same for all the merged blocks. In that case, the merged property
        labels will contains each of the entries of ``keys_to_move`` and then
        the current property labels. For example, using ``a=2, 3`` in
        ``keys_to_move``, and blocks with properties ``p=1, 2`` will result in
        ``a, p = (2, 1), (2, 2), (3, 1), (3, 2)``.

        The new sample labels will contains all of the merged blocks sample
        labels. The order of the samples is controlled by ``sort_samples``. If
        ``sort_samples`` is true, samples are re-ordered to keep them
        lexicographically sorted. Otherwise they are kept in the order in which
        they appear in the blocks.

        :param keys_to_move: description of the keys to move
        :param sort_samples: whether to sort the merged samples or keep them in
            the order in which they appear in the original blocks
        """
        keys_to_move = _normalize_keys_to_move(keys_to_move)
        self._lib.eqs_tensormap_keys_to_properties(
            self._ptr, keys_to_move._as_eqs_labels_t(), sort_samples
        )

    def keys_to_samples(
        self,
        keys_to_move: Union[str, List[str]],
        *,
        sort_samples=True,
    ):
        """
        Merge blocks with the same value for selected keys variables along the
        samples axis.

        The variables (names) of ``keys_to_move`` will be moved from the keys to
        the sample labels, and blocks with the same remaining keys variables
        will be merged together along the sample axis.

        If ``keys_to_move`` is a set of :py:class:`Labels`, it must be empty
        (``keys_to_move.shape[0] == 0``). The new sample labels will contain
        entries corresponding to the merged blocks' keys.

        The order of the samples is controlled by ``sort_samples``. If
        ``sort_samples`` is true, samples are re-ordered to keep them
        lexicographically sorted. Otherwise they are kept in the order in which
        they appear in the blocks.

        This function is only implemented if all merged block have the same
        property labels.

        :param keys_to_move: description of the keys to move
        :param sort_samples: whether to sort the merged samples or keep them in
            the order in which they appear in the original blocks
        """
        keys_to_move = _normalize_keys_to_move(keys_to_move)

        self._lib.eqs_tensormap_keys_to_samples(
            self._ptr, keys_to_move._as_eqs_labels_t(), sort_samples
        )

    def components_to_properties(self, variables: Union[str, List[str]]):
        """
        Move the given variables from the component labels to the property labels
        for each block.

        :param variables: name of the component variables to move to the properties
        """
        c_variables = _list_or_str_to_array_c_char(variables)

        self._lib.eqs_tensormap_components_to_properties(
            self._ptr, c_variables, c_variables._length_
        )

    @property
    def sample_names(self) -> List[str]:
        """Names of the sample labels for all blocks in this tensor map"""
        return self.block(0).samples.names

    @property
    def components_names(self) -> List[List[str]]:
        """Names of the component labels for all blocks in this tensor map"""
        return [c.names for c in self.block(0).components]

    @property
    def property_names(self) -> List[str]:
        """Names of the property labels for all blocks in this tensor map"""
        return self.block(0).properties.names


def _normalize_keys_to_move(keys_to_move: Union[str, List[str], Labels]):
    if isinstance(keys_to_move, str):
        keys_to_move = [keys_to_move]

    if isinstance(keys_to_move, list):
        for key in keys_to_move:
            assert isinstance(key, str)

        keys_to_move = Labels(
            names=keys_to_move,
            values=np.zeros((0, len(keys_to_move)), dtype=np.int32),
        )

    assert isinstance(keys_to_move, Labels)

    return keys_to_move


def _list_or_str_to_array_c_char(strings: Union[str, List[str]]):
    if isinstance(strings, str):
        strings = [strings]

    c_strings = ctypes.ARRAY(ctypes.c_char_p, len(strings))()
    for i, v in enumerate(strings):
        assert isinstance(v, str)
        c_strings[i] = v.encode("utf8")

    return c_strings
