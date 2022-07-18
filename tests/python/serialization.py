import hashlib
import os
import tempfile
import unittest

import numpy as np
from utils import test_tensor_map

import equistore.io
from equistore import EquistoreError, TensorMap

ROOT = os.path.dirname(__file__)


class TestIo(unittest.TestCase):
    def test_load(self):
        def check(tensor):
            self.assertIsInstance(tensor, TensorMap)
            self.assertEquals(
                tensor.keys.names,
                ("spherical_harmonics_l", "center_species", "neighbor_species"),
            )
            self.assertEquals(len(tensor.keys), 27)

            block = tensor.block(
                spherical_harmonics_l=2, center_species=6, neighbor_species=1
            )

            self.assertEquals(block.samples.names, ("structure", "center"))
            self.assertEquals(block.values.shape, (9, 5, 3))

            gradient = block.gradient("positions")
            self.assertEquals(gradient.samples.names, ("sample", "structure", "atom"))
            self.assertEquals(gradient.data.shape, (59, 3, 5, 3))

        path = os.path.join(ROOT, "..", "data.npz")

        try:
            tensor = equistore.io.load(path, use_numpy=False)
            check(tensor)
        except EquistoreError as e:
            self.assertEqual(
                str(e),
                "serialization format error: serialization was not enabled in equistore",
            )

        tensor = equistore.io.load(path, use_numpy=True)
        check(tensor)

    def test_save(self):
        def check_file(path, tensor):
            # check that the file loads fine with numpy
            data = np.load(path)
            self.assertEqual(len(data.keys()), 29)

            self.assertTrue(np.all(data["keys"] == tensor.keys))
            for i, (_, block) in enumerate(tensor):
                prefix = f"blocks/{i}/values"
                self.assertTrue(np.all(data[f"{prefix}/data"] == block.values))
                self.assertTrue(np.all(data[f"{prefix}/samples"] == block.samples))
                self.assertTrue(
                    np.all(data[f"{prefix}/components/0"] == block.components[0])
                )
                self.assertTrue(
                    np.all(data[f"{prefix}/properties"] == block.properties)
                )

                for parameter in block.gradients_list():
                    gradient = block.gradient(parameter)
                    prefix = f"blocks/{i}/gradients/{parameter}"
                    self.assertTrue(np.all(data[f"{prefix}/data"] == gradient.data))
                    self.assertTrue(
                        np.all(data[f"{prefix}/samples"] == gradient.samples)
                    )
                    self.assertTrue(
                        np.all(data[f"{prefix}/components/0"] == gradient.components[0])
                    )

        tmpfile = os.path.join(tempfile.gettempdir(), "serialize-test.npz")
        tensor = test_tensor_map()

        try:
            equistore.io.save(tmpfile, tensor, use_numpy=False)
            check_file(tmpfile, tensor)
        except EquistoreError as e:
            self.assertEqual(
                str(e),
                "serialization format error: serialization was not enabled in equistore",
            )

        equistore.io.save(tmpfile, tensor, use_numpy=True)
        check_file(tmpfile, tensor)


if __name__ == "__main__":
    unittest.main()
