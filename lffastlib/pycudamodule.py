import os
import re
import subprocess
from typing import List
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule
from pycuda import gpuarray

from lffastlib.timer import time_clock
from lffastlib.modify_cuda_code_programmatically import modify_cuda_code_on_the_fly
from lffastlib.pycutils import LabelingFunctions
from lffastlib.pycutils import CustomerFeatures



def compile_cuda_lib(source_code_cu: str, arch="60") -> List[str]:
    """
    Compiles CUDA code to cubin and extracts the list of compiled functions.

    Args:
        source_code_cu: Path to the CUDA source file

    Returns:
        List of compiled function names
    """
    # Step 1: Compile to cubin with specified architecture
    compile_cmd = [
        "nvcc", "-cubin", source_code_cu,
        "-o", f"{source_code_cu}.cubin",
        f"-arch=sm_{arch}",  "--generate-code", f"arch=compute_{arch},code=sm_{arch}"
    ]

    try:
        subprocess.run(compile_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e.stderr.decode()}")
        return []
    except FileNotFoundError:
        print("nvcc not found. Please install CUDA Toolkit")
        return []

    # Step 2: Run cuobjdump with --dump-section
    dump_cmd = [
        "cuobjdump",
        "-elf",
        f"{source_code_cu}.cubin"
    ]

    try:
        result = subprocess.run(dump_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"cuobjdump failed: {e.stderr}")
        return []
    except FileNotFoundError:
        print("cuobjdump not found")
        return []

    # Step 3: Parse function names
    functions = []
    for line in result.stdout.splitlines():

        if "CUDA_INFO" in line:
            #print(line)
            match = re.search(r'\.nv\.info\.([^\s]+)', line)
            if match:
                functions.append(match.group(1))

    return functions






class ComputeLabelingFunctionsCuda:
    """
    To init class you need to provide:

    Option1:
    file_lf_json - .json file with label function config
    df - dataframe (must have all features covered in file_lf_json, features not in file_lf_json will be ignored)
    column_id - specify column name (df) for ids (Specify None in case )
    column_label - specify column name (df) for labels (1 - SAR, 0 - noneSAR). Specify None in case nop labels

    """

    def __init__(self, *args, **kwargs):
        if len(args) == 2 and len(kwargs) in [1,2]:
            # First initialization option: file_lf_json, df, column_id, column_label
            file_lf_json, df = args
            column_id = kwargs.get('column_id')
            column_label = kwargs.get('column_label')

            if column_id is None:
                raise ValueError("'column_id' must be provided")

            self.LF = LabelingFunctions(file_lf_json)
            self.CF = CustomerFeatures(df, features=self.LF.features, column_id=column_id, column_label=column_label)


        else:
            raise ValueError("Invalid arguments passed to ComputeLabelingFunctions")

        assert self.LF.features == self.CF.features, "Mismatch between features in LF and CF object"

        self.lf_number = self.LF.lf_number
        self.features_number = self.CF.features_number
        self.customer_number = self.CF.customer_number


        self.lf_features_shape = self.LF.lf_features_shape.astype(np.int32)
        self.lf_features = self.LF.lf_features.astype(np.int32)
        self.lf_features_compare = self.LF.lf_features_compare.astype(np.int32)
        self.lf_features_thresholds = np.ascontiguousarray(self.LF.lf_features_thresholds).astype(np.float32)
        self.lf_outcome = self.LF.lf_outcome.astype(np.int32)

        self.customers_features = np.ascontiguousarray(self.CF.customers_features.flatten()).astype(np.float32)

        self.lf_features_start = np.cumsum(self.lf_features_shape, dtype=np.int32)
        self.lf_features_start = np.insert(self.lf_features_start, 0, 0)[:-1]

        self.customers_features_gpu = gpuarray.to_gpu(self.customers_features)
        self.lf_features_shape_gpu = gpuarray.to_gpu(self.lf_features_shape)
        self.lf_features_start_gpu = gpuarray.to_gpu(self.lf_features_start)
        self.lf_features_gpu = gpuarray.to_gpu(self.lf_features)
        self.lf_features_compare_gpu = gpuarray.to_gpu(self.lf_features_compare)
        self.lf_features_thresholds_gpu = gpuarray.to_gpu(self.lf_features_thresholds)
        self.lf_outcome_gpu = gpuarray.to_gpu(self.lf_outcome)

    def compute(self, output_type = "counts", kernel = "v1", block_size = 32, lf_features_shape_max = 10):

        results_type = 1
        if output_type == "matrix":
            results = np.zeros(self.customer_number*self.lf_number, dtype=np.int32)
            results_gpu = gpuarray.to_gpu(results)
            results_type = 0
        else:
            results = np.zeros(self.customer_number * 2, dtype=np.int32)
            results_gpu = gpuarray.to_gpu(results)
            results_type = 1


        print("self.customer_number", self.customer_number)
        print("self.features_number", self.features_number)
        print("self.lf_number", self.lf_number)


        start = pycuda.driver.Event()
        end = pycuda.driver.Event()



        # Create events for timing
        # Record start event
        start.record()

        if kernel == 'v2':

            lf_features_shape_max = lf_features_shape_max
            self.lf_features_2D = np.zeros(self.lf_number * lf_features_shape_max, dtype=np.int32)
            self.lf_features_compare_2D = np.zeros(self.lf_number * lf_features_shape_max, dtype=np.int32)
            self.lf_features_thresholds_2D = np.zeros(self.lf_number * lf_features_shape_max, dtype=np.float32)

            self.lf_features_2D_gpu = gpuarray.to_gpu(self.lf_features_2D)
            self.lf_features_compare_2D_gpu = gpuarray.to_gpu(self.lf_features_compare_2D)
            self.lf_features_thresholds_2D_gpu = gpuarray.to_gpu(self.lf_features_thresholds_2D)

            reshape_lf_features_kernel = self.kernel_functions["reshape_lf_features_kernel"]

            reshape_lf_features_kernel(
                np.int32(self.lf_number),
                self.lf_features_shape_gpu,
                self.lf_features_start_gpu,
                self.lf_features_gpu,
                self.lf_features_compare_gpu,
                self.lf_features_thresholds_gpu,

                self.lf_features_2D_gpu,
                self.lf_features_compare_2D_gpu,
                self.lf_features_thresholds_2D_gpu,

                block=(block_size, 1, 1),
                grid=(self.lf_number, 1, 1)

            )

            comp_label_customers_kernel = self.kernel_functions["comp_v2_label_customers"][self.features_number]

            comp_label_customers_kernel(
            np.int32(self.lf_number),
            self.lf_outcome_gpu,
            self.lf_features_shape_gpu,

            self.lf_features_2D_gpu,
            self.lf_features_compare_2D_gpu,
            self.lf_features_thresholds_2D_gpu,

            self.customers_features_gpu,
            np.int32(self.features_number),
            np.int32(self.customer_number),

            results_gpu,

            block=(block_size, 1, 1),
            grid=(self.customer_number,1,1)

            )



        elif kernel == 'v1':

            comp_label_customers_kernel = self.kernel_functions["comp_v1_label_customers"][self.features_number]

            comp_label_customers_kernel(
            np.int32(self.lf_number),
            self.lf_outcome_gpu,
            self.lf_features_shape_gpu,
            self.lf_features_start_gpu,
            self.lf_features_gpu,
            self.lf_features_compare_gpu,
            self.lf_features_thresholds_gpu,

            self.customers_features_gpu,
            np.int32(self.features_number),
            np.int32(self.customer_number),

            results_gpu,

            block=(block_size, 1, 1),
            grid=(self.customer_number,1,1)

            )
        elif kernel == 'v0':

            comp_label_customers_kernel = self.kernel_functions["comp_v0_label_customers"]

            comp_label_customers_kernel(
                np.int32(self.lf_number),
                self.lf_outcome_gpu,
                self.lf_features_shape_gpu,

                self.lf_features_gpu,
                self.lf_features_compare_gpu,
                self.lf_features_thresholds_gpu,

                self.customers_features_gpu,
                np.int32(self.features_number),
                np.int32(self.customer_number),

                results_gpu,

                block=(block_size, 1, 1),
                grid=(int((self.customer_number+block_size-1)/block_size), 1, 1)

            )

        end.record()
        end.synchronize()  # Wait for the kernel to finish

        # Calculate elapsed time in milliseconds
        elapsed_ms = start.time_till(end)
        print(f"Kernel execution time: {elapsed_ms:.3f} ms")

        results = results_gpu.get()
        return results

    @time_clock
    def get_cuda_lib(self, souce_code_cu):

        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        souce_code_cu = os.path.join(self.module_dir, souce_code_cu)
        #print(souce_code_cu)

        kernel_dict = {
            "comp_v0_label_customers": None,
            "comp_v1_label_customers": self.features_number,
            "comp_v2_label_customers": self.features_number,
            "reshape_lf_features_kernel": None
        }

        new_souce_code_cu = modify_cuda_code_on_the_fly(souce_code_cu, kernel_dict, metka="fly")

        # Get the compute capability of your GPU
        dev = pycuda.autoinit.device
        compute_capability = dev.compute_capability()
        arch_flag = f"{compute_capability[0]}{compute_capability[1]}"

        kernels = compile_cuda_lib(new_souce_code_cu, arch = arch_flag)

        # Load the compiled cubin
        with open(f"{new_souce_code_cu}.cubin", 'rb') as f:
            cubin = f.read()

        mod = cuda.module_from_buffer(cubin)

        # Get kernel functions
        kernel_functions = {}
        for name, param in kernel_dict.items():

            if param is None:
                for f in kernels:
                    if name in f:
                        kernel_functions[name] = mod.get_function(f)
                        break
            else:
                kernel_functions[name] = {}
                for f in kernels:

                    if name in f:
                        if str(param) in f:
                            kernel_functions[name][param] = mod.get_function(f)


        self.kernel_functions = kernel_functions







