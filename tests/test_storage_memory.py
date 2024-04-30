import os
import shutil
import datetime

import twosigma.memento as m
import tempfile

from twosigma.memento import Environment, ConfigurationRepository, FunctionCluster
from twosigma.memento.metadata import Memento, InvocationMetadata
from twosigma.memento.storage_memory import MemoryStorageBackend
from tests.test_storage_backend import StorageBackendTester


class TestStorageMemory(StorageBackendTester):
    """Class to test memory backend."""

    def setup_method(self):
        super().setup_method()
        self.original_env = m.Environment.get()
        self.base_path = tempfile.mkdtemp(prefix="memento_storage_memory_test")
        m.Environment.set({
            "name": "test1",
            "base_dir": self.base_path,
            "repos": [
                {
                    "name": "repo1",
                    "clusters": {
                        "cluster1": {
                            "name": "cluster1",
                            "storage": {
                                "type": "memory"
                            }
                        }
                    }
                }
            ]
        })
        self.cluster = m.Environment.get().get_cluster("cluster1")
        self.backend = self.cluster.storage

    def teardown_method(self):
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        m.Environment.set(self.original_env)
        super().teardown_method()

    # The rest of the test methods come from StorageBackendTester

    def test_make_url_for_result(self):
        # This test is not applicable for the memory store
        pass
