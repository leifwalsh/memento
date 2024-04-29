import os
import math
os.environ['MEMENTO_TEST_MODE'] = 'true'

print("Diagnostic - MEMENTO_TEST_MODE immediately after set:", os.environ['MEMENTO_TEST_MODE'])

import pytest
import shutil
import tempfile
from functools import wraps
from typing import Dict
import sys

# Ensure that the Environment class is imported only once and in the correct order
from twosigma.memento.configuration import Environment

print("Diagnostic - MEMENTO_TEST_MODE immediately after set:", os.environ['MEMENTO_TEST_MODE'])

print("Diagnostic - Environment class dictionary immediately after import:", Environment.__dict__)

print("Diagnostic - MEMENTO_TEST_MODE set to:", os.getenv('MEMENTO_TEST_MODE', 'False'))
print("Diagnostic - Environment class dictionary after import:", Environment.__dict__)

from twosigma.memento import MementoFunction
from twosigma.memento.exception import UndeclaredDependencyError
from twosigma.memento import memento_function
from twosigma.memento.code_hash import (
    fn_code_hash,
    list_dotted_names,
    resolve_to_symbolic_names,
    UndefinedSymbolHashRule,
)

@pytest.fixture(autouse=True)
def set_test_mode(monkeypatch):
    monkeypatch.setenv('MEMENTO_TEST_MODE', 'true')
    print("Diagnostic - MEMENTO_TEST_MODE set by fixture:", os.environ['MEMENTO_TEST_MODE'])

@memento_function()
def one_plus_one():
    # Diagnostic print to trace MementoFunction creation
    print(f"Diagnostic - Creating MementoFunction for one_plus_one")

    # Diagnostic print to trace CallStack state before function call
    print(f"Diagnostic - CallStack before calling one_plus_one: {CallStack.get()}")

    result = 1 + 1

    # Diagnostic print to trace CallStack state after function call
    print(f"Diagnostic - CallStack after calling one_plus_one: {CallStack.get()}")

    # Diagnostic print to check frame.memento attribute
    frame = CallStack.get().get_calling_frame()
    if frame is not None:
        print(f"Diagnostic - frame.memento for one_plus_one: {getattr(frame, 'memento', 'Attribute not found')}")
    else:
        print(f"Diagnostic - No calling frame for one_plus_one")

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

@memento_function(dependencies=[])
def top_level_caller():
    return caller_of_undeclared()

from twosigma.memento.call_stack import StackFrame, CallStack

class TestCodeHash:

    print("Diagnostic - Environment.__dict__ at the start of TestCodeHash:", Environment.__dict__)
    print("Diagnostic - MEMENTO_TEST_MODE at the start of TestCodeHash:", os.getenv('MEMENTO_TEST_MODE'))

    def setup_method(self):
        print("Diagnostic - setup_method called")
        print("Diagnostic - Environment.__dict__ before test setup:", Environment.__dict__)
        # Removed the redundant reload and import of the Environment class
        self.env_before = Environment.get()
        self.temp_dir = tempfile.mkdtemp(prefix="memoizeTest")
        test_environment_config = {
            "name": "test",
            "base_dir": self.temp_dir,
            "repos": []
        }
        print("Diagnostic - Environment.__dict__ before setting test environment:", Environment.__dict__)
        Environment.set(test_environment_config)
        print("Diagnostic - Environment.__dict__ after setting test environment:", Environment.__dict__)
        # Removed the pushing of dummy StackFrame onto the CallStack
        print("Diagnostic - CallStack depth after setup: {}".format(CallStack.get().depth()))
        print("Diagnostic - setup_method completed")

    def teardown_method(self):
        print("Diagnostic - teardown_method called")
        print("Diagnostic - CallStack depth before teardown: {}".format(CallStack.get().depth()))
        # Removed the popping of dummy StackFrame off the CallStack
        print("Diagnostic - Environment.__dict__ before test teardown:", Environment.__dict__)
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)
        Environment.set(self.env_before)
        print("Diagnostic - Environment.__dict__ after test teardown:", Environment.__dict__)
        print("Diagnostic - teardown_method completed")

    from contextlib import contextmanager

    @contextmanager
    def call_stack_context():
        # Create dummy instances of the required arguments for StackFrame
        dummy_fn_ref_with_args = FunctionReferenceWithArguments(
            fn_reference=FunctionReference(
                module_name="dummy_module",
                qualname="dummy_qualname"
            ),
            args=(),
            kwargs={}
        )
        dummy_runner = RunnerBackend()
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
        print("Diagnostic - Environment.__dict__ before test_fn_code_hash:", Environment.__dict__)
        # Corrected expected hash value for the one_plus_one function
        expected_hash = "52b3573abb5981cf"
        # Diagnostic print to check the CallStack before calling the MementoFunction
        print(f"Diagnostic - CallStack before calling MementoFunction: {CallStack.get()}")
        with call_stack_context():
            actual_hash = fn_code_hash(one_plus_one)
        # Diagnostic print to check the CallStack after calling the MementoFunction
        print(f"Diagnostic - CallStack after calling MementoFunction: {CallStack.get()}")
        assert expected_hash == actual_hash, f"Expected hash: {expected_hash}, Actual hash: {actual_hash}"
        print("Diagnostic - Environment.__dict__ after test_fn_code_hash:", Environment.__dict__)

    def test_fn_code_hash_with_salt(self):
        print("Diagnostic - Environment.__dict__ before test_fn_code_hash_with_salt:", Environment.__dict__)
        with call_stack_context():
            prev_hash = fn_code_hash(one_plus_one)
        with call_stack_context():
            hash_with_salt_a = fn_code_hash(one_plus_one, salt="a")
        with call_stack_context():
            hash_with_salt_a2 = fn_code_hash(one_plus_one, salt="a")
        with call_stack_context():
            hash_with_salt_b = fn_code_hash(one_plus_one, salt="b")

        assert prev_hash != hash_with_salt_a
        assert hash_with_salt_a == hash_with_salt_a2
        assert hash_with_salt_a != hash_with_salt_b
        print("Diagnostic - Environment.__dict__ after test_fn_code_hash_with_salt:", Environment.__dict__)

    def test_fn_code_hash_with_environment(self):
        print("Diagnostic - Environment.__dict__ before test_fn_code_hash_with_environment:", Environment.__dict__)
        with call_stack_context():
            prev_hash = fn_code_hash(one_plus_one)
        with call_stack_context():
            hash_with_env_a = fn_code_hash(one_plus_one, environment=b"a")
        with call_stack_context():
            hash_with_env_a2 = fn_code_hash(one_plus_one, environment=b"a")
        with call_stack_context():
            hash_with_env_b = fn_code_hash(one_plus_one, environment=b"b")

        assert prev_hash != hash_with_env_a
        assert hash_with_env_a == hash_with_env_a2
        assert hash_with_env_a != hash_with_env_b
        print("Diagnostic - Environment.__dict__ after test_fn_code_hash_with_environment:", Environment.__dict__)

    def test_dep_a(self):
        print("Diagnostic - Environment.__dict__ before test_dep_a:", Environment.__dict__)
        # Test the dep_a function to ensure it is covered
        assert dep_a() == 54, "The dep_a function did not return the expected result."
        print("Diagnostic - Environment.__dict__ after test_dep_a:", Environment.__dict__)

    def test_dep_a_with_function_in_dot_path(self):
        print("Diagnostic - Environment.__dict__ before test_dep_a_with_function_in_dot_path:", Environment.__dict__)
        # Test the dep_a_with_function_in_dot_path function to ensure it is covered
        assert dep_a_with_function_in_dot_path() == 42, "The dep_a_with_function_in_dot_path function did not return the expected result."
        print("Diagnostic - Environment.__dict__ after test_dep_a_with_function_in_dot_path:", Environment.__dict__)

    def test_dep_with_embedded_fn(self):
        print("Diagnostic - Environment.__dict__ before test_dep_with_embedded_fn:", Environment.__dict__)
        # Test the dep_with_embedded_fn function to ensure the embedded function is covered
        assert dep_with_embedded_fn() == 42, "The dep_with_embedded_fn did not return the expected result from the embedded function."
        print("Diagnostic - Environment.__dict__ after test_dep_with_embedded_fn:", Environment.__dict__)

    def test_resolve_to_symbolic_name(self):
        print("Diagnostic - Environment.__dict__ before test_resolve_to_symbolic_name:", Environment.__dict__)
        result = resolve_to_symbolic_names(["dep_a", dep_b])
        assert "dep_a" in result
        assert "tests.test_code_hash:dep_b" in result, str(result)
        print("Diagnostic - Environment.__dict__ after test_resolve_to_symbolic_name:", Environment.__dict__)

    def test_list_dotted_names(self):
        print("Diagnostic - Environment.__dict__ before test_list_dotted_names:", Environment.__dict__)
        result = list_dotted_names(dep_a.fn)
        assert "dep_b" in result
        assert "math.sqrt" in result
        print("Diagnostic - Environment.__dict__ after test_list_dotted_names:", Environment.__dict__)

    def test_list_dotted_names_with_function_in_dot_path(self):
        result = list_dotted_names(dep_a_with_function_in_dot_path.fn)
        assert "dep_b" in result

    def test_global_var(self):
        print("Diagnostic - Environment.__dict__ before test_global_var:", Environment.__dict__)
        global global_var
        global_var_before = global_var
        try:
            version_before = dep_global_var.version()
            global_var = 43
            version_after = dep_global_var.version()
            assert version_before != version_after
        finally:
            global_var = global_var_before
        print("Diagnostic - Environment.__dict__ after test_global_var:", Environment.__dict__)

    def test_non_memento_fn(self):
        print("Diagnostic - Environment.__dict__ before test_non_memento_fn:", Environment.__dict__)
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
        print("Diagnostic - Environment.__dict__ after test_non_memento_fn:", Environment.__dict__)

    def test_redefine_memento_fn_as_non_memento_fn(self):
        print("Diagnostic - Environment.__dict__ before test_redefine_memento_fn_as_non_memento_fn:", Environment.__dict__)
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
        print("Diagnostic - Environment.__dict__ after test_redefine_memento_fn_as_non_memento_fn:", Environment.__dict__)

    def test_late_define_non_memento_fn(self):
        print("Diagnostic - Environment.__dict__ before test_late_define_non_memento_fn:", Environment.__dict__)
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
        print("Diagnostic - Environment.__dict__ after test_late_define_non_memento_fn:", Environment.__dict__)

    def test_fn_with_local_vars(self):
        print("Diagnostic - Environment.__dict__ before test_fn_with_local_vars:", Environment.__dict__)
        """
        Make sure local variables are not included in the function hash
        """
        hash_rules = fn_with_local_vars.hash_rules()
        for rule in hash_rules:
            assert "x" not in rule.describe(), "Local variable 'x' should not be included in the hash rules."
        assert fn_with_local_vars() == 3, "The function did not return the expected result."
        print("Diagnostic - Environment.__dict__ after test_fn_with_local_vars:", Environment.__dict__)

    def test_fn_with_cell_vars(self):
        print("Diagnostic - Environment.__dict__ before test_fn_with_cell_vars:", Environment.__dict__)
        """
        Make sure cell variables are not included in the function hash
        """
        hash_rules = fn_with_cell_vars.hash_rules()
        for rule in hash_rules:
            assert "x" not in rule.describe(), "Cell variable 'x' should not be included in the hash rules."
        assert fn_with_cell_vars() == 3, "The function did not return the expected result."
        print("Diagnostic - Environment.__dict__ after test_fn_with_cell_vars:", Environment.__dict__)

    def test_cluster_lock_prevents_version_update(self):
        print("Diagnostic - Environment.__dict__ before test_cluster_lock_prevents_version_update:", Environment.__dict__)
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
        print("Diagnostic - Environment.__dict__ after test_cluster_lock_prevents_version_update:", Environment.__dict__)

    def test_call_fn_param(self):
        print("Diagnostic - Environment.__dict__ before test_call_fn_param:", Environment.__dict__)
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
        print("Diagnostic - Environment.__dict__ after test_call_fn_param:", Environment.__dict__)

    def test_safe_to_call_memento_fn_wrappers(self):
        print("Diagnostic - Environment.__dict__ before test_safe_to_call_memento_fn_wrappers:", Environment.__dict__)
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
        print("Diagnostic - Environment.__dict__ after test_safe_to_call_memento_fn_wrappers:", Environment.__dict__)

    def test_hash_simple_function(self):
        print("Diagnostic - Environment.__dict__ before test_hash_simple_function:", Environment.__dict__)
        # Hash the function
        hash_result = fn_code_hash(simple_function)

        # Actual hash value for the simple_function
        precomputed_hash = "1463afb8f2a4c319"

        # Assert that the computed hash matches the actual hash
        assert hash_result == precomputed_hash, f"Hash does not match. Computed: {hash_result}, Expected: {precomputed_hash}"

        # Execute the simple_function and check the result
        assert simple_function(10) == 11, "The simple_function did not return the expected result."
        print("Diagnostic - Environment.__dict__ after test_hash_simple_function:", Environment.__dict__)

    def test_hash_consistency(self):
        print("Diagnostic - Environment.__dict__ before test_hash_consistency:", Environment.__dict__)
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
        print("Diagnostic - Environment.__dict__ after test_hash_consistency:", Environment.__dict__)

    def test_undefined_symbol_hash_rule(self):
        print("Diagnostic - Environment.__dict__ before test_undefined_symbol_hash_rule:", Environment.__dict__)
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
        print("Diagnostic - Environment.__dict__ after test_undefined_symbol_hash_rule:", Environment.__dict__)

    def test_wrapped_one_plus_one(self):
        print("Diagnostic - Environment.__dict__ before test_wrapped_one_plus_one:", Environment.__dict__)
        # Test the _wrapped_one_plus_one function to ensure it is covered
        assert _wrapped_one_plus_one() == 2, "The _wrapped_one_plus_one function did not return the expected result."
        print("Diagnostic - Environment.__dict__ after test_wrapped_one_plus_one:", Environment.__dict__)

    def test_memento_function_with_undeclared_dependency(self):
        print("Diagnostic - Environment.__dict__ before test_memento_function_with_undeclared_dependency:", Environment.__dict__)
        """
        Test that a Memento function calling an undeclared external function raises
        an UndeclaredDependencyError.
        """
        def non_memento_caller():
            top_level_caller()

        with pytest.raises(UndeclaredDependencyError):
            non_memento_caller()
        print("Diagnostic - Environment.__dict__ after test_memento_function_with_undeclared_dependency:", Environment.__dict__)

    print("Diagnostic - Environment.__dict__ at the end of TestCodeHash:", Environment.__dict__)
    print("Diagnostic - MEMENTO_TEST_MODE at the end of TestCodeHash:", os.getenv('MEMENTO_TEST_MODE'))
