import os
import math
from unittest.mock import Mock
os.environ['MEMENTO_TEST_MODE'] = 'true'

import pytest
import shutil
import tempfile
from functools import wraps
from typing import Dict
import sys

# Ensure that the Environment class is imported only once and in the correct order
from twosigma.memento.configuration import Environment

from twosigma.memento import MementoFunction
from twosigma.memento.exception import UndeclaredDependencyError
from twosigma.memento import memento_function

from twosigma.memento.metadata import Memento, InvocationMetadata

from twosigma.memento.code_hash import (
    fn_code_hash,
    list_dotted_names,
    resolve_to_symbolic_names,
    UndefinedSymbolHashRule,
)
from twosigma.memento.call_stack import FunctionReferenceWithArguments, RunnerBackend, RecursiveContext
from twosigma.memento.reference import FunctionReference

@pytest.fixture(autouse=True)
def set_test_mode(monkeypatch):
    monkeypatch.setenv('MEMENTO_TEST_MODE', 'true')

@memento_function()
def one_plus_one():
    result = 1 + 1
    return result


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

@memento_function()
def mock_memento_fn():
    pass

def top_level_caller():
    return caller_of_undeclared()

from twosigma.memento.call_stack import StackFrame, CallStack

class TestCodeHash:

    def setup_method(self):
        self.env_before = Environment.get()
        self.temp_dir = tempfile.mkdtemp(prefix="memoizeTest")
        test_environment_config = {
            "name": "test",
            "base_dir": self.temp_dir,
            "repos": []
        }
        Environment.set(test_environment_config)

    def teardown_method(self):
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
        Environment.set(self.env_before)

    from contextlib import contextmanager

    @staticmethod
    @contextmanager
    def call_stack_context():
        dummy_fn_ref_with_args = FunctionReferenceWithArguments(
            fn_reference=FunctionReference(
                memento_fn=mock_memento_fn,
                cluster_name="dummy_cluster",
                module_name="dummy_module",
                function_name="dummy_qualname"
            ),
            args=(),
            kwargs={}
        )
        dummy_runner = Mock(spec=RunnerBackend)
        dummy_recursive_context = RecursiveContext()

        # Initialize StackFrame with the dummy instances
        frame = StackFrame(
            fn_reference_with_args=dummy_fn_ref_with_args,
            runner=dummy_runner,
            recursive_context=dummy_recursive_context
        )
        CallStack.get().push_frame(frame)
        try:
            yield
        finally:
            CallStack.get().pop_frame()

    def test_fn_code_hash(self):
        expected_hash = "8be6343f7b497448"
        with TestCodeHash.call_stack_context():
            actual_hash = fn_code_hash(one_plus_one)
        assert expected_hash == actual_hash, f"Expected hash: {expected_hash}, Actual hash: {actual_hash}"

    def test_fn_code_hash_with_salt(self):
        with TestCodeHash.call_stack_context():
            prev_hash = fn_code_hash(one_plus_one)
        with TestCodeHash.call_stack_context():
            hash_with_salt_a = fn_code_hash(one_plus_one, salt="a")
        with TestCodeHash.call_stack_context():
            hash_with_salt_a2 = fn_code_hash(one_plus_one, salt="a")
        with TestCodeHash.call_stack_context():
            hash_with_salt_b = fn_code_hash(one_plus_one, salt="b")

        assert prev_hash != hash_with_salt_a
        assert hash_with_salt_a == hash_with_salt_a2
        assert hash_with_salt_a != hash_with_salt_b

    def test_fn_code_hash_with_environment(self):
        with TestCodeHash.call_stack_context():
            prev_hash = fn_code_hash(one_plus_one)
        with TestCodeHash.call_stack_context():
            hash_with_env_a = fn_code_hash(one_plus_one, environment=b"a")
        with TestCodeHash.call_stack_context():
            hash_with_env_a2 = fn_code_hash(one_plus_one, environment=b"a")
        with TestCodeHash.call_stack_context():
            hash_with_env_b = fn_code_hash(one_plus_one, environment=b"b")

        assert prev_hash != hash_with_env_a
        assert hash_with_env_a == hash_with_env_a2
        assert hash_with_env_a != hash_with_env_b

    def test_dep_a(self):
        print("Before calling dep_a, result_type is set to:", type(dep_a.result_type).__name__)
        assert dep_a() == 54, "The dep_a function did not return the expected result."
        print("After calling dep_a, result_type is set to:", type(dep_a.result_type).__name__)

    def test_dep_a_with_function_in_dot_path(self):
        assert dep_a_with_function_in_dot_path() == 42, "The dep_a_with_function_in_dot_path function did not return the expected result."

    def test_dep_with_embedded_fn(self):
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
            _floating_fn = _non_memento_fn_1
            assert {
                dep_b
            } == dep_floating_fn.dependencies().transitive_memento_fn_dependencies()
            assert _floating_fn() == 42, "The _non_memento_fn_1 did not return the expected result."

            _floating_fn = _non_memento_fn_2
            assert {
                dep_a,
                dep_b,
            } == dep_floating_fn.dependencies().transitive_memento_fn_dependencies()
            assert _floating_fn() == 54, "The _non_memento_fn_2 did not return the expected result."
        finally:
            _floating_fn = _non_memento_fn_1

    def test_redefine_memento_fn_as_non_memento_fn(self):
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
        hash_rules = fn_with_local_vars.hash_rules()
        for rule in hash_rules:
            assert "x" not in rule.describe(), "Local variable 'x' should not be included in the hash rules."
        assert fn_with_local_vars() == 3, "The function did not return the expected result."

    def test_fn_with_cell_vars(self):
        hash_rules = fn_with_cell_vars.hash_rules()
        for rule in hash_rules:
            assert "x" not in rule.describe(), "Cell variable 'x' should not be included in the hash rules."
        assert fn_with_cell_vars() == 3, "The function did not return the expected result."

    def test_cluster_lock_prevents_version_update(self):
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
        wrapped_fn = _memento_fn_wrapper(one_plus_one)
        assert wrapped_fn() == 2, "The wrapped function did not return the expected result."

        result = (
            fn_calls_wrapped_one_plus_one.dependencies().transitive_memento_fn_dependencies()
        )
        assert {_wrapped_one_plus_one.__wrapped__} == result

        assert fn_calls_wrapped_one_plus_one() == 2, "fn_calls_wrapped_one_plus_one did not return the expected result."

    def test_hash_simple_function(self):
        hash_result = fn_code_hash(simple_function)

        precomputed_hash = "1463afb8f2a4c319"

        assert hash_result == precomputed_hash, f"Hash does not match. Computed: {hash_result}, Expected: {precomputed_hash}"

        assert simple_function(10) == 11, "The simple_function did not return the expected result."

    def test_hash_consistency(self):
        def consistent_function(x):
            return x * 2

        first_hash = fn_code_hash(consistent_function)
        second_hash = fn_code_hash(consistent_function)

        assert first_hash == second_hash, f"Hashes are not consistent. First: {first_hash}, Second: {second_hash}"

        assert consistent_function(10) == 20, "The consistent_function did not return the expected result."

    def test_undefined_symbol_hash_rule(self):
        global undefined_global_var

        try:
            del undefined_global_var
            hash_rules = fn_with_undefined_global.hash_rules()
            assert any(isinstance(rule, UndefinedSymbolHashRule) for rule in hash_rules), "UndefinedSymbolHashRule not created for undefined global variable."
            with pytest.raises(NameError) as exc_info:
                fn_with_undefined_global()
            assert "name 'undefined_global_var' is not defined" in str(exc_info.value)
        finally:
            undefined_global_var = 42

    def test_wrapped_one_plus_one(self):
        assert _wrapped_one_plus_one() == 2, "The _wrapped_one_plus_one function did not return the expected result."

    def test_memento_function_with_undeclared_dependency(self):
        def non_memento_caller():
            top_level_caller()

        with pytest.raises(UndeclaredDependencyError):
            non_memento_caller()
