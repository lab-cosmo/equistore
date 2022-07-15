import copy
import ctypes
import gc
from typing import Generator, List, Tuple

from ._c_api import c_uintptr_t, eqs_array_t, eqs_labels_t
from ._c_lib import _get_library
from .data import (
    Array,
    ArrayWrapper,
    eqs_array_to_python_object,
    eqs_array_was_allocated_by_python,
)
from .labels import Labels
from .status import _check_pointer


class TensorBlock:
    """
    Basic building block for a tensor map.

    A single block contains a n-dimensional :py:class:`equistore.data.Array`,
    and n sets of :py:class:`Labels` (one for each dimension). The first
    dimension is the *samples* dimension, the last dimension is the *properties*
    dimension. Any intermediate dimension is called a *component* dimension.

    Samples should be used to describe *what* we are representing, while
    properties should contain information about *how* we are representing it.
    Finally, components should be used to describe vectorial or tensorial
    components of the data.

    A block can also contain gradients of the values with respect to a variety
    of parameters. In this case, each gradient has a separate set of samples,
    and possibly components but share the same property labels as the values.
    """

    def __init__(
        self,
        values: Array,
        samples: Labels,
        components: List[Labels],
        properties: Labels,
    ):
        """
        :param values: array containing the data for this block
        :param samples: labels describing the samples (first dimension of the
            array)
        :param components: labels describing the components (second dimension of
            the array). This is set to :py:func:`Labels.single` when dealing
            with scalar/invariant values.
        :param properties: labels describing the samples (third dimension of the
            array)
        """
        self._lib = _get_library()

        # keep a reference to the values in the block to prevent GC
        self._values = ArrayWrapper(values)
        self._gradients = []

        components_array = ctypes.ARRAY(eqs_labels_t, len(components))()
        for i, component in enumerate(components):
            components_array[i] = component._as_eqs_labels_t()

        self._ptr = self._lib.eqs_block(
            self._values.eqs_array,
            samples._as_eqs_labels_t(),
            components_array,
            len(components_array),
            properties._as_eqs_labels_t(),
        )
        self._owning = True
        self._parent = None
        _check_pointer(self._ptr)

    @staticmethod
    def _from_ptr(ptr, parent, owning):
        """
        create a block from a pointer, either owning its data (new block as a
        copy of an existing one) or not (block inside a :py:class:`TensorMap`)
        """
        _check_pointer(ptr)
        obj = TensorBlock.__new__(TensorBlock)
        obj._lib = _get_library()
        obj._ptr = ptr
        obj._owning = owning
        obj._values = None
        obj._gradients = []
        # keep a reference to the parent object (usually a TensorMap) to
        # prevent it from beeing garbage-collected & removing this block
        obj._parent = parent
        return obj

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_ptr") and hasattr(self, "_owning"):
            if self._owning:
                self._lib.eqs_block_free(self._ptr)

    def __deepcopy__(self, _memodict={}):
        # Temporarily disable garbage collection to ensure the temporary
        # ArrayWrapper instance created in _eqs_array_copy will still be
        # available below
        if gc.isenabled():
            gc.disable()
            reset_gc = True
        else:
            reset_gc = False

        new_ptr = self._lib.eqs_block_copy(self._ptr)
        copy = TensorBlock._from_ptr(new_ptr, parent=None, owning=True)

        # Keep references to the arrays in this block if the arrays were
        # allocated by Python
        raw_values = _get_raw_array(self._lib, copy._ptr, "values")
        if eqs_array_was_allocated_by_python(raw_values):
            copy._values = eqs_array_to_python_object(raw_values)
        else:
            copy._values = None

        for parameter in self.gradients_list():
            raw_array = _get_raw_array(self._lib, copy._ptr, parameter)
            if eqs_array_was_allocated_by_python(raw_array):
                copy._gradients.append(eqs_array_to_python_object(raw_array))
            else:
                pass

        if reset_gc:
            gc.enable()

        return copy

    def copy(self) -> "TensorBlock":
        """
        Get a deep copy of this block
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        s = "TensorBlock\n"
        s += "    samples: ["
        for sname in self.samples.names[:-1]:
            s += "'" + sname + "', "
        s += "'" + self.samples.names[-1] + "']"
        s += "\n"
        s += "    component: ["
        for ic in self.components:
            for name in ic.names[:]:
                s += "'" + name + "', "
        if len(self.components) > 0:
            s = s[:-2]
        s += "]\n"
        s += "    properties: ["
        for name in self.properties.names[:-1]:
            s += "'" + name + "', "
        s += "'" + self.properties.names[-1] + "']\n"
        s += "    gradients: "
        if len(self.gradients_list()) > 0:
            s += "["
            for gr in self.gradients_list()[:-1]:
                s += "'{}', ".format(gr)
            s += "'{}']".format(self.gradients_list()[-1])
        else:
            s += "no"

        return s

    @property
    def values(self) -> Array:
        """
        Access the values for this block.

        The array type depends on how the block was created. Currently, numpy
        ``ndarray`` and torch ``Tensor`` are supported.
        """

        raw_array = _get_raw_array(self._lib, self._ptr, "values")
        return eqs_array_to_python_object(raw_array, parent=self).array

    @property
    def samples(self) -> Labels:
        """
        Access the sample :py:class:`Labels` for this block.

        The entries in these labels describe the first dimension of the
        ``values`` array.
        """
        return self._labels(0)

    @property
    def components(self) -> List[Labels]:
        """
        Access the component :py:class:`Labels` for this block.

        The entries in these labels describe intermediate dimensions of the
        ``values`` array.
        """
        n_components = len(self.values.shape) - 2

        result = []
        for axis in range(n_components):
            result.append(self._labels(axis + 1))

        return result

    @property
    def properties(self) -> Labels:
        """
        Access the property :py:class:`Labels` for this block.

        The entries in these labels describe the last dimension of the
        ``values`` array. The properties are guaranteed to be the same for
        values and gradients in the same block.
        """
        property_axis = len(self.values.shape) - 1
        return self._labels(property_axis)

    def _labels(self, axis) -> Labels:
        result = eqs_labels_t()
        self._lib.eqs_block_labels(self._ptr, "values".encode("utf8"), axis, result)
        return Labels._from_eqs_labels_t(result, parent=self)

    def gradient(self, parameter: str) -> "Gradient":
        """
        Get the gradient of the ``values`` in this block with respect to
        the given ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)
        """
        if not self.has_gradient(parameter):
            raise ValueError(
                f"this block does not contain gradient with respect to {parameter}"
            )
        return Gradient(self, parameter)

    def add_gradient(
        self,
        parameter: str,
        data: Array,
        samples: Labels,
        components: List[Labels],
    ):
        """
        Add a set of gradients with respect to ``parameters`` in this block.

        :param parameter: add gradients with respect to this ``parameter`` (e.g.
            ``positions``, ``cell``, ...)
        :param data: the gradient array, of shape ``(gradient_samples,
            components, properties)``, where the properties labels are the same
            as the values' properties labels.
        :param samples: labels describing the gradient samples
        :param components: labels describing the gradient components
        """
        if self._parent is not None:
            raise ValueError(
                "can not add gradient on this block since it is a view inside "
                "a TensorMap"
            )

        data = ArrayWrapper(data)
        self._gradients.append(data)

        components_array = ctypes.ARRAY(eqs_labels_t, len(components))()
        for i, component in enumerate(components):
            components_array[i] = component._as_eqs_labels_t()

        self._lib.eqs_block_add_gradient(
            self._ptr,
            parameter.encode("utf8"),
            data.eqs_array,
            samples._as_eqs_labels_t(),
            components_array,
            len(components_array),
        )

    def gradients_list(self) -> List[str]:
        """Get a list of all gradients defined in this block."""
        parameters = ctypes.POINTER(ctypes.c_char_p)()
        count = c_uintptr_t()
        self._lib.eqs_block_gradients_list(self._ptr, parameters, count)

        result = []
        for i in range(count.value):
            result.append(parameters[i].decode("utf8"))

        return result

    def has_gradient(self, parameter: str) -> bool:
        """
        Check if this block contains gradient information with respect to the
        given ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)
        """
        return parameter in self.gradients_list()

    def gradients(self) -> Generator[Tuple[str, "Gradient"], None, None]:
        """Get an iterator over all gradients defined in this block."""
        for parameter in self.gradients_list():
            yield (parameter, self.gradient(parameter))


class Gradient:
    """
    Proxy class allowing to access the information associated with a gradient
    inside a block.
    """

    def __init__(self, block: TensorBlock, name: str):
        self._lib = _get_library()

        self._block = block
        self._name = name

    def __repr__(self) -> str:
        s = "Gradient TensorBlock \n"
        s += "parameter: '{}'\n".format(self._name)
        s += "samples: ["
        for sname in self.samples.names[:-1]:
            s += "'" + sname + "', "
        s += "'" + self.samples.names[-1] + "']"
        s += "\n"
        s += "component: ["
        for ic in self.components:
            for name in ic.names[:]:
                s += "'" + name + "', "
        if len(self.components) > 0:
            s = s[:-2]
        s += "]\n"
        s += "properties: ["
        for name in self.properties.names[:-1]:
            s += "'" + name + "', "
        s += "'" + self.properties.names[-1] + "']"

        return s

    @property
    def data(self) -> Array:
        """
        Access the data for this gradient.

        The array type depends on how the block was created. Currently, numpy
        ``ndarray`` and torch ``Tensor`` are supported.
        """

        raw_array = _get_raw_array(self._lib, self._block._ptr, self._name)
        return eqs_array_to_python_object(raw_array, parent=self).array

    @property
    def samples(self) -> Labels:
        """
        Access the sample :py:class:`Labels` for this gradient.

        The entries in these labels describe the first dimension of the ``data``
        array.
        """
        return self._labels(0)

    @property
    def components(self) -> List[Labels]:
        """
        Access the component :py:class:`Labels` for this gradient.

        The entries in these labels describe intermediate dimensions of the
        ``data`` array.
        """
        n_components = len(self.data.shape) - 2

        result = []
        for axis in range(n_components):
            result.append(self._labels(axis + 1))

        return result

    @property
    def properties(self) -> Labels:
        """
        Access the property :py:class:`Labels` for this gradient.

        The entries in these labels describe the last dimension of the ``data``
        array. The properties are guaranteed to be the same for values and
        gradients in the same block.
        """
        property_axis = len(self.data.shape) - 1
        return self._labels(property_axis)

    def _labels(self, axis) -> Labels:
        result = eqs_labels_t()
        self._lib.eqs_block_labels(
            self._block._ptr, self._name.encode("utf8"), axis, result
        )
        return Labels._from_eqs_labels_t(result, parent=self._block)


def _get_raw_array(lib, block_ptr, name) -> eqs_array_t:
    data = eqs_array_t()
    lib.eqs_block_data(block_ptr, name.encode("utf8"), data)
    return data
