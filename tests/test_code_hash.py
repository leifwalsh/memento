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

import os
import importlib
import twosigma.memento.configuration as configuration
importlib.reload(configuration)
from twosigma.memento.configuration import Environment

import math
import shutil
import tempfile
from functools import wraps
from typing import Dict
import sys

import pytest

from twosigma.memento import MementoFunction
from twosigma.memento.exception import UndeclaredDependencyError
from twosigma.memento import memento_function
from twosigma.memento.code_hash import (
    fn_code_hash,
    list_dotted_names,
    resolve_to_symbolic_names,
    UndefinedSymbolHashRule,
)

@memento_function()
def one_plus_one():
    return 1 + 1


@memento_function()
def dep_a():
    return dep_b() + math.sqrt(144)


@memento_function()
def dep_a_with_function_in_dot_path():
    return dep_b()


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


# Define a simple function with no dependencies
def simple_function(x):
    return x + 1


@memento_function
def fn_with_undefined_global():
    return undefined_global_var

# Global variable for testing undefined symbol hash rule
undefined_global_var = None

def undeclared_external_function():
    return "This function is not declared as a dependency."

@memento_function(dependencies=[])
def memento_fn_calling_undeclared():
    return undeclared_external_function()

@memento_function(dependencies=[])
def caller_of_undeclared():
    return memento_fn_calling_undeclared()

@memento_function(dependencies=[])
def top_level_caller():
    return caller_of_undeclared()

class TestCodeHash:

    def setup_method(self):
        import os
        import importlib
        import tempfile

        # Set the MEMENTO_TEST_MODE environment variable to 'true'
        os.environ['MEMENTO_TEST_MODE'] = 'true'

        # Reload the configuration module to ensure the latest version is used
        import twosigma.memento.configuration as configuration
        importlib.reload(configuration)
        from twosigma.memento.configuration import Environment

        print("Checking if 'is_test_mode' is in Environment: ", 'is_test_mode' in dir(Environment))
        self.env_before = Environment.get()
        self.env_dir = tempfile.mkdtemp(prefix="memoizeTest")
        env_file = "{}/env.json".format(self.env_dir)
        with open(env_file, "w") as f:
            print("""{"name": "test"}""", file=f)
        Environment.set(env_file)

    def teardown_method(self):
        shutil.rmtree(self.env_dir)
        Environment.set(self.env_before)

    def test_fn_code_hash(self):
        # Corrected expected hash value for the one_plus_one function
        expected_hash = "52b3573abb5981cf"
        actual_hash = fn_code_hash(one_plus_one)
        assert expected_hash == actual_hash, f"Expected hash: {expected_hash}, Actual hash: {actual_hash}"

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

    def test_dep_a(self):
        # Test the dep_a function to ensure it is covered
        assert dep_a() == 54, "The dep_a function did not return the expected result."

    def test_dep_a_with_function_in_dot_path(self):
        # Test the dep_a_with_function_in_dot_path function to ensure it is covered
        assert dep_a_with_function_in_dot_path() == 42, "The dep_a_with_function_in_dot_path function did not return the expected result."

    def test_dep_with_embedded_fn(self):
        # Test the dep_with_embedded_fn function to ensure the embedded function is covered
        assert dep_with_embedded_fn() == 42, "The dep_with_embedded_fn did not return the expected result from the embedded function."

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
        assert "dep_b" in result

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
            # Ensure that dep_floating_fn depends on _floating_fn which is _non_memento_fn_1
            _floating_fn = _non_memento_fn_1
            assert {
                dep_b
            } == dep_floating_fn.dependencies().transitive_memento_fn_dependencies()
            assert _floating_fn() == 42, "The _non_memento_fn_1 did not return the expected result."

            # Change _floating_fn to _non_memento_fn_2 and ensure the dependency is updated
            _floating_fn = _non_memento_fn_2
            assert {
                dep_a,
                dep_b,
            } == dep_floating_fn.dependencies().transitive_memento_fn_dependencies()
            assert _floating_fn() == 54, "The _non_memento_fn_2 did not return the expected result."
        finally:
            _floating_fn = _non_memento_fn_1

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
        hash_rules = fn_with_local_vars.hash_rules()
        for rule in hash_rules:
            assert "x" not in rule.describe(), "Local variable 'x' should not be included in the hash rules."
        assert fn_with_local_vars() == 3, "The function did not return the expected result."

    def test_fn_with_cell_vars(self):
        """
        Make sure cell variables are not included in the function hash
        """
        hash_rules = fn_with_cell_vars.hash_rules()
        for rule in hash_rules:
            assert "x" not in rule.describe(), "Cell variable 'x' should not be included in the hash rules."
        assert fn_with_cell_vars() == 3, "The function did not return the expected result."

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
            fn_test_call_fn_param(dep_a)
            fn_test_call_fn_param(dep_with_embedded_fn)
            fn_test_call_dict_with_list_with_fn_param({"fn": [one_plus_one]})
            fn_test_call_dict_with_list_with_fn_param({"fn": [dep_a]})
            fn_test_call_dict_with_list_with_fn_param({"fn": [dep_with_embedded_fn]})
        except UndeclaredDependencyError:
            pytest.fail("Not expected to raise UndeclaredDependencyError")

    def test_safe_to_call_memento_fn_wrappers(self):
        """
        Check that wrappers of memento functions correctly register a dependency on the
        function.
        """
        # Directly test the _memento_fn_wrapper by creating a new wrapped function instance
        wrapped_fn = _memento_fn_wrapper(one_plus_one)
        assert wrapped_fn() == 2, "The wrapped function did not return the expected result."

        # Check that the dependencies are correctly registered
        result = (
            fn_calls_wrapped_one_plus_one.dependencies().transitive_memento_fn_dependencies()
        )
        # noinspection PyUnresolvedReferences
        assert {_wrapped_one_plus_one.__wrapped__} == result

        # Explicitly test the return value of fn_calls_wrapped_one_plus_one
        assert fn_calls_wrapped_one_plus_one() == 2, "fn_calls_wrapped_one_plus_one did not return the expected result."

    def test_hash_simple_function(self):
        # Hash the function
        hash_result = fn_code_hash(simple_function)

        # Actual hash value for the simple_function
        precomputed_hash = "1463afb8f2a4c319"

        # Assert that the computed hash matches the actual hash
        assert hash_result == precomputed_hash, f"Hash does not match. Computed: {hash_result}, Expected: {precomputed_hash}"

        # Execute the simple_function and check the result
        assert simple_function(10) == 11, "The simple_function did not return the expected result."

    def test_hash_consistency(self):
        # Define a simple function with no dependencies
        def consistent_function(x):
            return x * 2

        # Hash the function twice
        first_hash = fn_code_hash(consistent_function)
        second_hash = fn_code_hash(consistent_function)

        # Assert that both hashes are the same, indicating consistency
        assert first_hash == second_hash, f"Hashes are not consistent. First: {first_hash}, Second: {second_hash}"

        # Execute the consistent_function and check the result
        assert consistent_function(10) == 20, "The consistent_function did not return the expected result."

    def test_undefined_symbol_hash_rule(self):
        """
        Test that an undefined symbol triggers the creation of an UndefinedSymbolHashRule.
        """
        global undefined_global_var

        try:
            del undefined_global_var
            hash_rules = fn_with_undefined_global.hash_rules()
            assert any(isinstance(rule, UndefinedSymbolHashRule) for rule in hash_rules), "UndefinedSymbolHashRule not created for undefined global variable."
            # Attempt to execute fn_with_undefined_global, which should raise a NameError
            with pytest.raises(NameError) as exc_info:
                fn_with_undefined_global()
            assert "name 'undefined_global_var' is not defined" in str(exc_info.value)
        finally:
            undefined_global_var = 42  # Restore to prevent side effects on other tests

    def test_wrapped_one_plus_one(self):
        # Test the _wrapped_one_plus_one function to ensure it is covered
        assert _wrapped_one_plus_one() == 2, "The _wrapped_one_plus_one function did not return the expected result."

    def test_memento_function_with_undeclared_dependency(self):
        """
        Test that a Memento function calling an undeclared external function raises
        an UndeclaredDependencyError.
        """
        def non_memento_caller():
            top_level_caller()

        with pytest.raises(UndeclaredDependencyError):
            non_memento_caller()
