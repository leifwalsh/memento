# Copyright (c) 2023 Two Sigma Investments, LP.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import shutil
import tempfile
from functools import wraps
from typing import Dict

import pytest

from twosigma.memento import MementoFunction
from twosigma.memento.exception import UndeclaredDependencyError
from twosigma.memento import memento_function, Environment
from twosigma.memento.code_hash import (
    fn_code_hash,
    list_dotted_names,
    resolve_to_symbolic_names,
)


@memento_function()
def one_plus_one():
    return 1 + 1


@memento_function()
def dep_a():
    return dep_b() + math.sqrt(144)


@memento_function()
def dep_a_with_function_in_dot_path():
    return dep_b.ignore_result().call()


@memento_function()
def dep_b():
    return 42


# Global variable depended-on by a memento function. Should affect hash.
global_var = 42


@memento_function()
def dep_global_var():
    return global_var


@memento_function()
def dep_floating_fn():
    return _floating_fn()


def _non_memento_fn_1():
    return dep_b()


def _non_memento_fn_2():
    return dep_a()


def _non_memento_fn_3():
    return 0


_floating_fn = _non_memento_fn_1


@memento_function()
def dep_with_embedded_fn():
    def embedded_fn():
        return dep_b()

    return embedded_fn()


@memento_function
def fn_with_local_vars():
    x = 3
    return x


@memento_function
def fn_with_cell_vars():
    x = 3

    def inner():
        y = x
        return y

    return inner()


@memento_function
def fn_test_call_fn_param(fn: MementoFunction):
    return fn()


@memento_function
def fn_test_call_dict_with_list_with_fn_param(d: Dict):
    return d["fn"][0]()


def _memento_fn_wrapper(fn: MementoFunction):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


@_memento_fn_wrapper
@memento_function
def _wrapped_one_plus_one():
    return 1 + 1


@memento_function
def fn_calls_wrapped_one_plus_one():
    return _wrapped_one_plus_one()


class TestCodeHash:

    def setup_method(self):
        import sys  # Ensure sys is imported to access sys.modules
        print("Loaded modules before setup:", list(sys.modules.keys()))
        print("Environment before setup:", Environment.get())
        self.env_before = Environment.get()
        self.env_dir = tempfile.mkdtemp(prefix="memoizeTest")
        env_file = "{}/env.json".format(self.env_dir)
        with open(env_file, "w") as f:
            print("""{"name": "test"}""", file=f)
        Environment.set(env_file)
        print("Environment after setup:", Environment.get())

    def teardown_method(self):
        import sys  # Ensure sys is imported to access sys.modules
        print("Loaded modules before teardown:", list(sys.modules.keys()))
        print("Environment before teardown:", Environment.get())
        shutil.rmtree(self.env_dir)
        Environment.set(self.env_before)
        print("Environment after teardown:", Environment.get())
        print("Loaded modules after teardown:", list(sys.modules.keys()))

    @pytest.mark.needs_canonical_version
    def test_fn_code_hash(self):
        expected_hash = "e4306c39c214411e"
        assert expected_hash == fn_code_hash(one_plus_one)

    def test_fn_code_hash_with_salt(self):
        prev_hash = fn_code_hash(one_plus_one)
        hash_with_salt_a = fn_code_hash(one_plus_one, salt="a")
        hash_with_salt_a2 = fn_code_hash(one_plus_one, salt="a")
        hash_with_salt_b = fn_code_hash(one_plus_one, salt="b")

        assert prev_hash != hash_with_salt_a
        assert hash_with_salt_a == hash_with_salt_a2
        assert hash_with_salt_a != hash_with_salt_b

    def test_fn_code_hash_with_environment(self):
        prev_hash = fn_code_hash(one_plus_one)
        hash_with_env_a = fn_code_hash(one_plus_one, environment=b"a")
        hash_with_env_a2 = fn_code_hash(one_plus_one, environment=b"a")
        hash_with_env_b = fn_code_hash(one_plus_one, environment=b"b")

        assert prev_hash != hash_with_env_a
        assert hash_with_env_a == hash_with_env_a2
        assert hash_with_env_a != hash_with_env_b

    def test_resolve_to_symbolic_name(self):
        result = resolve_to_symbolic_names(["dep_a", dep_b])
        assert "dep_a" in result
        assert "tests.test_code_hash:dep_b" in result, str(result)

    def test_list_dotted_names(self):
        result = list_dotted_names(dep_a.fn)
        assert "dep_b" in result
        assert "math.sqrt" in result

    def test_list_dotted_names_with_function_in_dot_path(self):
        result = list_dotted_names(dep_a_with_function_in_dot_path.fn)
        assert "dep_b.ignore_result.call" in result

    def test_global_var(self):
        global global_var
        global_var_before = global_var
        try:
            version_before = dep_global_var.version()
            global_var = 43
            version_after = dep_global_var.version()
            assert version_before != version_after
        finally:
            global_var = global_var_before

    def test_non_memento_fn(self):
        global _floating_fn

        try:
            assert {
                dep_b
            } == dep_floating_fn.dependencies().transitive_memento_fn_dependencies()

            version_before = dep_floating_fn.version()
            _floating_fn = _non_memento_fn_2
            version_after = dep_floating_fn.version()
            assert version_before != version_after

            assert {
                dep_a,
                dep_b,
            } == dep_floating_fn.dependencies().transitive_memento_fn_dependencies()
        finally:
            _floating_fn = _non_memento_fn_1

    def test_dep_with_embedded_fn(self):
        assert {
            dep_b
        } == dep_with_embedded_fn.dependencies().transitive_memento_fn_dependencies()

    def test_redefine_memento_fn_as_non_memento_fn(self):
        """
        memento fn f calls memento fn g and g is redefined as a non-memento fn.
        Ensure Memento realizes g has changed.

        """
        global _floating_fn

        try:
            _floating_fn = dep_b
            v1 = dep_floating_fn.version()
            assert 42 == dep_floating_fn()
            _floating_fn = _non_memento_fn_3
            v2 = dep_floating_fn.version()
            assert 0 == dep_floating_fn()
            assert v1 != v2
        finally:
            _floating_fn = _non_memento_fn_1

    def test_late_define_non_memento_fn(self):
        """
        memento fn f calls function g which is undefined at the time of the definition of f.
        g is later defined as a non-memento fn. Ensure Memento changes the version of f.

        """
        global _floating_fn

        try:
            del _floating_fn
            v1 = dep_floating_fn.version()
            with pytest.raises(NameError):
                dep_floating_fn()

            _floating_fn = _non_memento_fn_3
            v2 = dep_floating_fn.version()
            assert v1 != v2
            assert 0 == dep_floating_fn()
        finally:
            _floating_fn = _non_memento_fn_1

    def test_fn_with_local_vars(self):
        """
        Make sure local variables are not included in the function hash

        """
        assert not any(
            "UndefinedSymbol;x" in r.describe() for r in fn_with_local_vars.hash_rules()
        )

    def test_fn_with_cell_vars(self):
        """
        Make sure cell variables are not included in the function hash

        """
        assert not any(
            "UndefinedSymbol;x" in r.describe() for r in fn_with_cell_vars.hash_rules()
        )

    def test_cluster_lock_prevents_version_update(self):
        """
        Test that locking the cluster prevents the version of a function from updating, even if
        the function changes.

        """
        global global_var

        prev_value = global_var
        cluster = Environment.get().get_cluster(None)
        try:
            cluster.locked = True
            v1 = dep_floating_fn.version()
            assert prev_value == dep_global_var()
            global_var = prev_value + 1
            v2 = dep_floating_fn.version()
            assert (
                prev_value == dep_global_var()
            )  # Should be memoized from previous call
            assert v1 == v2
        finally:
            global_var = prev_value
            cluster.locked = False

    def test_call_fn_param(self):
        """
        Test that calling a function that is passed as a parameter does not cause an
        `UndeclaredDependencyError`.

        """
        try:
            fn_test_call_fn_param(one_plus_one)
            fn_test_call_dict_with_list_with_fn_param({"fn": [one_plus_one]})
        except UndeclaredDependencyError:
            pytest.fail("Not expected to raise UndeclaredDependencyError")

    def test_safe_to_call_memento_fn_wrappers(self):
        """
        Check that wrappers of memento functions correctly register a dependency on the
        function.

        """
        result = (
            fn_calls_wrapped_one_plus_one.dependencies().transitive_memento_fn_dependencies()
        )
        # noinspection PyUnresolvedReferences
        assert {_wrapped_one_plus_one.__wrapped__} == result

    def test_hash_simple_function(self):
        import os
        import sys  # Import sys to access sys.flags and sys.modules

        # Define a simple function with no dependencies
        def simple_function(x):
            return x + 1

        # Print the current environment variables for debugging purposes
        print("Current environment variables:")
        for key, value in os.environ.items():
            print(f"{key}: {value}")

        # Print the sys.flags for debugging purposes
        print("Python sys.flags:", sys.flags)

        # Print the loaded modules for debugging purposes
        print("Loaded modules:", list(sys.modules.keys()))

        # Print the co_flags attribute of the simple_function code object for debugging purposes
        print("co_flags of simple_function:", simple_function.__code__.co_flags)

        # Additional print statements to check co_flags of other functions in this test class
        print("co_flags of consistent_function:", consistent_function.__code__.co_flags)

        # Hash the function
        hash_result = fn_code_hash(simple_function)

        # Print the computed hash for debugging purposes
        print(f"Computed hash: {hash_result}")

        # Actual hash value for the simple_function
        precomputed_hash = "200960dc9a77feaf"

        # Assert that the computed hash matches the actual hash
        assert hash_result == precomputed_hash, f"Hash does not match. Computed: {hash_result}, Expected: {precomputed_hash}"

    def test_hash_consistency(self):
        # Define a simple function with no dependencies
        def consistent_function(x):
            return x * 2

        # Hash the function twice
        first_hash = fn_code_hash(consistent_function)
        second_hash = fn_code_hash(consistent_function)

        # Assert that both hashes are the same, indicating consistency
        assert first_hash == second_hash, f"Hashes are not consistent. First: {first_hash}, Second: {second_hash}"

    def test_compute_actual_hash_for_simple_function(self):
        # Define a simple function with no dependencies
        def simple_function(x):
            return x + 1

        # Compute the hash for the simple function
        actual_hash = fn_code_hash(simple_function)
        print(f"Actual computed hash for simple_function: {actual_hash}")
