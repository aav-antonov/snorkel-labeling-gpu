"""
This script is a comprehensive test for comparing different implementations of Labeling Functions (LFs) computation:
1. Native Snorkel implementation (Python)
2. C multithreading implementation
3. Three different CUDA GPU implementations (v0, v1, v2 kernels)

The test:
- Generates random test cases with varying parameters:
  - Number of customers
  - Number of features
  - Number of labeling functions
  - Size range of labeling functions
- For each test case:
  - Computes results using all implementations
  - Validates that all implementations produce identical results
  - Uses assertions to verify result equality

The purpose is to ensure:
1. All implementations are functionally equivalent
2. The CUDA kernels produce correct results matching the reference Snorkel implementation
3. The code works across different input sizes and configurations
"""

import os
import numpy as np

from lffastlib.pycmodule import ComputeLabelingFunctions
from lffastlib.pycudamodule import ComputeLabelingFunctionsCuda
from lffastlib.generate_case_rand import GeneratorLF
from lffastlib.snorkeljson import SnorkelLFJsonConverter


if __name__ == "__main__":


    cpu_cores = os.cpu_count()  # Returns total logical CPU cores
    print(f"Available CPU cores: {cpu_cores}")

    ## Generate random case )
    customer_number = [1, 10, 100]
    features_number = [50, 100]
    lf_number = [1, 10, 100]
    lf_size_range = [5, 50]

    # max number of features in lf, used only for cuda kernel="v2";
    lf_features_shape_max = lf_size_range[1];

    converter = SnorkelLFJsonConverter()

    for customer_n in customer_number:
        for features_n in features_number:
            for lf_n in lf_number:

                #Generate random sample
                input ={}
                input['customer_number'] = customer_n
                input['features_number'] = features_n
                input['lf_number'] = lf_n
                input['lf_size_range'] = lf_size_range

                GLF = GeneratorLF(**input)

                # Compute by native snorkel: results_snorkel
                converter.json_to_lfs(GLF.lf_json)
                results_snorkel = converter.apply_lfs_to_df(GLF.df)

                # Compute by C multithreading: results_cpu
                CLFCpu = ComputeLabelingFunctions(GLF.lf_json, GLF.df, column_id='id')
                results_cpu = CLFCpu.compute(threads=cpu_cores)

                # Check results_cpu is identical to results_snorkel
                assert np.array_equal(results_cpu, results_snorkel), "Arrays are not equal!"
                
                
                # Init PyCuda ComputeClass
                CLFCuda = ComputeLabelingFunctionsCuda(GLF.lf_json, GLF.df, column_id='id')
                CLFCuda.get_cuda_lib("cuda_code/comp.cu")

                # Compute by naive kernel = "v0" : results_gpu_v0
                results_gpu_v0 = CLFCuda.compute(kernel="v0", block_size=32)

                # Check results_gpu_v0 is identical to results_snorkel
                assert np.array_equal(results_gpu_v0, results_snorkel), "Arrays are not equal!"

                # Compute by opimize kernel = "v1" : results_gpu_v1
                results_gpu_v1 = CLFCuda.compute(kernel="v1", block_size=32)

                # Check results_gpu_v1 is identical to results_snorkel
                assert np.array_equal(results_gpu_v1, results_snorkel), "Arrays are not equal!"

                # Compute by opimize kernel = "v2" : results_gpu_v2
                results_gpu_v2 = CLFCuda.compute(kernel="v2", block_size=32, lf_features_shape_max = lf_features_shape_max)

                # Check results_gpu_v2 is identical to results_snorkel
                assert np.array_equal(results_gpu_v2, results_snorkel), "Arrays are not equal!"
