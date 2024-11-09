import unittest
from autoop.core.storage import LocalStorage, NotFoundError
import random
import tempfile
import os


class TestStorage(unittest.TestCase):
    """
    Unit test class for testing the LocalStorage functionality.
    """

    def setUp(self) -> None:
        """
        Sets up the test environment by creating
        a temporary LocalStorage instance.
        """
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self) -> None:
        """Tests that the LocalStorage instance is initialized correctly."""
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self) -> None:
        """
        Tests storing and retrieving data,
        ensuring that saved data matches loaded data.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.sep}path"

        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)

        otherkey = f"test{os.sep}otherpath"
        try:
            self.storage.load(otherkey)
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_delete(self) -> None:
        """
        Tests deleting data,
        confirming it is no longer accessible after deletion.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.sep}path"

        self.storage.save(test_bytes, key)
        self.storage.delete(key)

        # Attempt to load deleted data and confirm it raises NotFoundError
        try:
            self.assertIsNone(self.storage.load(key))
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_list(self) -> None:
        """
        Tests listing all keys in a directory,
        verifying that the stored keys match expected keys.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [f"test{os.sep}{random.randint(0, 100)}"
                       for _ in range(10)]

        for key in random_keys:
            self.storage.save(test_bytes, key)

        keys = self.storage.list("test")
        keys = [f"{os.sep}".join(key.split(f"{os.sep}")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
