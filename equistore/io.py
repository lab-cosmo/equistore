import warnings

import numpy as np

from ._c_lib import _get_library
from .block import TensorBlock
from .data import _is_numpy_array, _is_torch_array
from .labels import Labels
from .tensor import TensorMap


def load(path: str, use_numpy=False) -> TensorMap:
    """
    Load a previously saved :py:class:`equistore.TensorMap` from the given path.

    :py:class:`equistore.TensorMap` are serialized using numpy's ``.npz``
    format, i.e. a ZIP file without compression (storage method is ``STORED``),
    where each file is stored as a ``.npy`` array. See the C API documentation
    for more information on the format.

    :param path: path of the file to load
    :param use_numpy: should we use numpy or the native implementation? Numpy
        should be able to process more dtypes than the native implementation,
        which is limited to float64, but the native implementation is usually
        faster than going through numpy.
    """
    if use_numpy:
        return _read_npz(path)
    else:
        lib = _get_library()
        return TensorMap._from_ptr(lib.eqs_tensormap_load(path.encode("utf8")))


def save(path: str, tensor: TensorMap, use_numpy=False):
    """Save the given :py:class:`equistore.TensorMap` to a file at ``path``.

    :py:class:`equistore.TensorMap` are serialized using numpy's ``.npz``
    format, i.e. a ZIP file without compression (storage method is ``STORED``),
    where each file is stored as a ``.npy`` array. See the C API documentation
    for more information on the format.

    :param path: path of the file where to save the data
    :param tensor: tensor to save
    :param use_numpy: should we use numpy or the native implementation? Numpy
        should be able to process more dtypes than the native implementation,
        which is limited to float64, but the native implementation is usually
        faster than going through numpy.
    """
    if not path.endswith(".npz"):
        path += ".npz"
        warnings.warn(f"adding '.npz' extension, the file will be saved at '{path}'")

    if use_numpy:
        all_entries = _tensor_map_to_dict(tensor)
        np.savez(path, **all_entries)
    else:
        lib = _get_library()
        lib.eqs_tensormap_save(path.encode("utf8"), tensor._ptr)


def _array_to_numpy(array):
    if _is_numpy_array(array):
        return array
    elif _is_torch_array(array):
        return array.cpu().numpy()
    else:
        raise ValueError("unknown array type passed to `equistore.io.save`")


def _tensor_map_to_dict(tensor_map):
    result = {"keys": tensor_map.keys}

    for block_i, (_, block) in enumerate(tensor_map):
        prefix = f"blocks/{block_i}/values"
        result[f"{prefix}/data"] = _array_to_numpy(block.values)
        result[f"{prefix}/samples"] = block.samples
        for i, component in enumerate(block.components):
            result[f"{prefix}/components/{i}"] = component
        result[f"{prefix}/properties"] = block.properties

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            prefix = f"blocks/{block_i}/gradients/{parameter}"
            result[f"{prefix}/data"] = _array_to_numpy(gradient.data)
            result[f"{prefix}/samples"] = gradient.samples
            for i, component in enumerate(gradient.components):
                result[f"{prefix}/components/{i}"] = component

    return result


def _labels_from_npz(data):
    names = data.dtype.names
    return Labels(names=names, values=data.view(dtype=np.int32).reshape(-1, len(names)))


def _read_npz(path):
    dictionary = np.load(path)

    keys = _labels_from_npz(dictionary["keys"])
    blocks = []

    gradient_parameters = []
    for block_i in range(len(keys)):
        prefix = f"blocks/{block_i}/values"
        data = dictionary[f"{prefix}/data"]

        samples = _labels_from_npz(dictionary[f"{prefix}/samples"])
        components = []
        for i in range(len(data.shape) - 2):
            components.append(_labels_from_npz(dictionary[f"{prefix}/components/{i}"]))

        properties = _labels_from_npz(dictionary[f"{prefix}/properties"])

        block = TensorBlock(data, samples, components, properties)

        if block_i == 0:
            prefix = f"blocks/{block_i}/gradients/"
            for name in dictionary.keys():
                if name.startswith(prefix) and name.endswith("/data"):
                    gradient_parameters.append(name[len(prefix) : -len("/data")])

        for parameter in gradient_parameters:
            prefix = f"blocks/{block_i}/gradients/{parameter}"
            data = dictionary[f"{prefix}/data"]

            samples = _labels_from_npz(dictionary[f"{prefix}/samples"])
            components = []
            for i in range(len(data.shape) - 2):
                components.append(
                    _labels_from_npz(dictionary[f"{prefix}/components/{i}"])
                )

            block.add_gradient(parameter, data, samples, components)

        blocks.append(block)

    return TensorMap(keys, blocks)
