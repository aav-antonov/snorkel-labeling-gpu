import time

from lffastlib.pycmodule import ComputeLabelingFunctions
from lffastlib.pycudamodule import ComputeLabelingFunctionsCuda
from lffastlib.generate_case_rand import GeneratorLF
from lffastlib.snorkeljson import SnorkelLFJsonConverter

import numpy as np
import pandas as pd
import os



if __name__ == "__main__":

    cpu_cores = os.cpu_count()  # Returns total logical CPU cores
    print(f"Available CPU cores: {cpu_cores}")

    customer_number = [1000000]
    lf_number = [100000]
    features_number = [400]
    lf_size_range = [10, 10]
    
    lf_features_shape_max = lf_size_range[1];

    execution_time_data = {}
    
    execution_time_data[f"C_cpucores_{cpu_cores}"] = []
    execution_time_data["GPU_kernel_v0"] = []
    execution_time_data["GPU_kernel_v1"] = []
    execution_time_data["GPU_kernel_v2"] = []

    for customer_n in customer_number:
        for features_n in features_number:
            for lf_n in lf_number:

                input = {}
                input['customer_number'] = customer_n
                input['features_number'] = features_n
                input['lf_number'] = lf_n
                input['lf_size_range'] = lf_size_range

                GLF = GeneratorLF(**input)

                

                # CPU timing
                CLFCpu = ComputeLabelingFunctions(GLF.lf_json, GLF.df, column_id='id')
                start_time = time.perf_counter()
                results_cpu = CLFCpu.compute(threads=cpu_cores)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                print(f"CPU: {execution_time:.3f}s")

                execution_time_data[f"C_cpucores_{cpu_cores}"].append([customer_n, lf_n, features_n, execution_time])

                # GPU timing
                CLFCuda = ComputeLabelingFunctionsCuda(GLF.lf_json, GLF.df, column_id='id')
                CLFCuda.get_cuda_lib("cuda_code/comp.cu")

                # v0
                start_time = time.perf_counter()
                results_gpu_v0 = CLFCuda.compute(kernel="v0", block_size=32)
                end_time = time.perf_counter()
                execution_time_v0 = end_time - start_time
                print(f"GPU v0: {execution_time_v0:.3f}s")
                execution_time_data["GPU_kernel_v0"].append([customer_n, lf_n, features_n, execution_time_v0])
                assert np.array_equal(results_gpu_v0, results_cpu), "Arrays are not equal!"

                # v1
                start_time = time.perf_counter()
                results_gpu_v1 = CLFCuda.compute(kernel="v1", block_size=32)
                end_time = time.perf_counter()
                execution_time_v1 = end_time - start_time
                print(f"GPU v1: {execution_time_v1:.3f}s")
                execution_time_data["GPU_kernel_v1"].append([customer_n, lf_n, features_n, execution_time_v1])
                assert np.array_equal(results_gpu_v1, results_cpu), "Arrays are not equal!"

                # v2
                start_time = time.perf_counter()
                results_gpu_v2 = CLFCuda.compute(kernel="v2", block_size=32, lf_features_shape_max = lf_features_shape_max)
                end_time = time.perf_counter()
                execution_time_v2 = end_time - start_time
                print(f"GPU v2: {execution_time_v2:.3f}s")
                execution_time_data["GPU_kernel_v2"].append([customer_n, lf_n, features_n, execution_time_v2])
                assert np.array_equal(results_gpu_v2, results_cpu), "Arrays are not equal!"


    # Collect all results into a flattened list with the method as a field
    rows = []
    for method, entries in execution_time_data.items():
        for entry in entries:
            rows.append({
                'Method': method,
                'customer_n': entry[0],
                'lf_n': entry[1],
                'features_n': entry[2],
                'execution_time(s)': entry[3]
            })
    
    # Sort for pretty output (optional)
    rows = sorted(rows, key=lambda x: (x['customer_n'], x['lf_n'], x['features_n'], x['Method']))
    

    
    # (Optional) Also save as markdown, as before
    with open("results_large_scale.md", "w") as f:
        f.write("| Method | customer_n | lf_n | features_n | execution_time(s) |\n")
        f.write("|--------|------------|------|------------|-------------------|\n")
        for row in rows:
            f.write(f"| {row['Method']} | {row['customer_n']} | {row['lf_n']} | {row['features_n']} | {row['execution_time(s)']:.5f} |\n")
    
    print("Results saved to results_large_scale.md")
