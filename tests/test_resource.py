import os
import shutil
import tempfile
from pathlib import Path

import twosigma.memento as m
from twosigma.memento import memento_function, Memento  # noqa: F401
from twosigma.memento.resource_function import file_resource


_called = False


@memento_function
def fn_test_depends_on_file(path: str) -> int:
    file_resource(path)
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0


@memento_function
def fn_test_sample_1(path: str) -> int:
    return fn_test_depends_on_file(path)


class TestResource:
    """Class to test memento resources."""

    def setup_method(self):
        global _called
        self.env_before = m.Environment.get()
        self.env_dir = tempfile.mkdtemp(prefix="resourceTest")
        m.Environment.set({
            "name": "test",
            "base_dir": self.env_dir
        })
        _called = False

    def teardown_method(self):
        shutil.rmtree(self.env_dir)
        m.Environment.set(self.env_before)

    def test_file_resource_handle(self):
        """
        Tests the attribute of a file resource handle are set correctly.

        """
        file_path = Path(self.env_dir).joinpath("1.txt")
        with file_path.open("w") as f:
            f.write("foo")
        mtime = str(int(round(os.path.getmtime(file_path) * 1000)))

        handle = file_resource(str(file_path))
        assert "file" == handle.resource_type
        assert file_path.as_uri() == handle.url
        assert mtime == handle.version

    def test_resource_dependency(self):
        """
        Test that a function can depend on a resource and that the resource is present in the
        memento.

        """
        file_path = Path(self.env_dir).joinpath("3.txt")
        with file_path.open("w") as f:
            f.write("foo")

        assert 3 == fn_test_depends_on_file(str(file_path))
        # Now, test that the memoized version is retrieved properly
        assert 3 == fn_test_depends_on_file(str(file_path))
        memento = fn_test_depends_on_file.memento(str(file_path))  # type: Memento
        resources = memento.invocation_metadata.resources
        assert 1 == len(resources)
        assert file_path.as_uri() == resources[0].url
