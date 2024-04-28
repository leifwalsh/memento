import json

# Function to extract modules from the provided text
def extract_modules(text):
    start = text.find("Loaded modules: [") + len("Loaded modules: [")
    end = text.find("]", start)
    modules_str = text[start:end]
    # Split the string by comma and strip whitespace to create a list of modules
    modules_list = [module.strip() for module in modules_str.split(',')]
    return modules_list

# Function to compare two lists of modules and find differences
def compare_module_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.symmetric_difference(set2))

# Read the output file
with open("/home/ubuntu/full_outputs/_Retrieve_the_conten_1714281013.248676.txt", "r") as file:
    output_text = file.read()

# Split the output text to separate the manual and test run module lists
split_texts = output_text.split("attr_values: [1,")
manual_text = split_texts[0]
test_run_text = split_texts[-1]  # Take the last occurrence for the test run modules

# Extract the lists of loaded modules
manual_modules = extract_modules(manual_text)
test_run_modules = extract_modules(test_run_text)

# Compare the lists and find differences
module_differences = compare_module_lists(manual_modules, test_run_modules)

# Output the differences to a JSON file
with open("module_differences.json", "w") as file:
    json.dump(module_differences, file, indent=4)

print("Module differences have been written to module_differences.json")
