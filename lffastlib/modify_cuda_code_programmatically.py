import re
import os

def modify_cuda_code_on_the_fly(file_path, functions_dict, metka="fly"):
    """
    Modify CUDA code by inserting/updating template instantiations for multiple functions.
    Automatically parses parameter lists from kernel definitions.

    Args:
        file_path: Path to the original .cu file
        functions_dict: Dictionary of {function_name_str: k} pairs where:
                       - function_name_str is the template function name
                       - k is the template parameter to insert/update
        metka: Suffix for the modified file
    Returns:
        Path to the modified file
    """
    # Read the original file
    with open(file_path, 'r') as f:
        content = f.read()

    # First parse all kernel definitions to get parameter lists
    kernel_pattern = re.compile(
        r'template\s*<\s*int\s+\w+\s*>\s*'  # Template part
        r'__global__\s+void\s+'              # Kernel declaration
        r'(\w+)'                             # Function name (captured)
        r'\s*<\s*\w+\s*>\s*'                 # Template argument
        r'\((.*?)\)\s*\{',                   # Parameter list (captured)
        re.DOTALL
    )

    # Build dictionary of function parameters
    functions_param_dict = {}
    for match in kernel_pattern.finditer(content):
        func_name = match.group(1)
        params = match.group(2).strip()
        # Clean up parameters
        params = re.sub(r'/\*.*?\*/', '', params)  # Remove comments
        params = re.sub(r'//.*$', '', params)
        params = re.sub(r'\s+', ' ', params)       # Normalize whitespace
        functions_param_dict[func_name] = params.strip()

    # Process each function to update/create template instantiations
    for function_name_str, new_k in functions_dict.items():
        if new_k is None:
            continue

        # Try to find existing instantiation first
        existing_pattern = re.compile(
            r'^\s*template\s+__global__\s+void\s+' +
            re.escape(function_name_str) +
            r'<\d+>\(.*?\);',
            re.MULTILINE
        )

        def update_k(match):
            old_line = match.group(0)
            return re.sub(r'<\d+>', f'<{new_k}>', old_line)

        # If exists, update the k value
        if existing_pattern.search(content):
            content = existing_pattern.sub(update_k, content)
        else:
            # Else create new instantiation using parsed parameters
            param_str = functions_param_dict.get(function_name_str, "")
            new_template = f"template __global__ void {function_name_str}<{new_k}>({param_str});\n"
            content += '\n' + new_template

    # Create new file path
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_{metka}{ext}"

    # Write the modified content
    with open(new_file_path, 'w') as f:
        f.write(content)

    return new_file_path