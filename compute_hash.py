from twosigma.memento.code_hash import fn_code_hash
import sys

def simple_function(x):
    return x + 1

# Print the loaded modules for debugging purposes
print("Loaded modules:", list(sys.modules.keys()))

# Print the co_flags attribute of the simple_function code object for debugging purposes
print("co_flags:", simple_function.__code__.co_flags)

print(fn_code_hash(simple_function))
