from twosigma.memento.code_hash import fn_code_hash
import sys

def simple_function(x):
    return x + 1

# Print the loaded modules for debugging purposes
print("Loaded modules:", list(sys.modules.keys()))

print(fn_code_hash(simple_function))
