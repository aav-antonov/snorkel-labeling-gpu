import ctypes
import numpy as np
import os



def compile_c(dir_c, files_c):
    
    object_files = []

    # Step 1: Compile each C file into an object file
    for file_c in files_c[:-1]:  # All but the last file
        c_file_path = os.path.join(dir_c, f"{file_c}")
        path_parts = c_file_path.split(os.sep)
        path_parts.insert(-1, 'exec')
        c_obj_path = os.sep.join(path_parts)
        
        obj_file = c_obj_path.replace('.c', '.o')
        obj_file = os.path.join(f"{dir_c}/exec", obj_file)
        compile_command = f"g++ -c -o {obj_file} -fPIC {c_file_path}"
        compilation_result = os.system(compile_command)
        
        if compilation_result != 0:
            print(f"ERROR: Compilation of {file_c} failed with exit code", compilation_result)
            exit(1)
        
        object_files.append(obj_file)

    # Step 2: Compile the last C file and link with the other object files
    
    last_c_file = os.path.join(dir_c,f"{files_c[-1]}")
    path_parts = last_c_file.split(os.sep)
    path_parts.insert(-1, 'exec')
    exec_file = os.sep.join(path_parts)
    exec_file = exec_file.replace('.c', '.so')
    
       
    obj_files_str = ' '.join(object_files)
    
    compile_command = f"g++ -shared -o {exec_file} -fPIC {last_c_file} {obj_files_str} -lpthread"
    compilation_result = os.system(compile_command)

    if compilation_result == 0:
        print(f"Compilation successful. {exec_file} created.")
    else:
        print("ERROR: Compilation failed with exit code", compilation_result)
        exit(1)
    
    return ctypes.CDLL(exec_file)