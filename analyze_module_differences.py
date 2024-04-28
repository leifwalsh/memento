import json

# List of modules known to affect bytecode generation or execution environment
known_modules = [
    # Add known modules here
]

# Load the differences from the comparison output
with open('module_differences.json', 'r') as file:
    differences = json.load(file)

# Filter the differences for known modules
affecting_modules = {
    'manual_not_test': [mod for mod in differences['manual_not_test'] if mod in known_modules],
    'test_not_manual': [mod for mod in differences['test_not_manual'] if mod in known_modules]
}

# Output the filtered differences
print('Modules in manual computation not in test run that are known to affect bytecode generation or execution environment:', affecting_modules['manual_not_test'])
print('Modules in test run not in manual computation that are known to affect bytecode generation or execution environment:', affecting_modules['test_not_manual'])
