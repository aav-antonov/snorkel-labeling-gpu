# High-Performance Labeling Functions with C and CUDA 


This repository aims to provide significantly faster implementations for computing specific types of labeling functions, compared to Snorkel's native Python implementations. This speedup is achieved through the use of C multithreading and multiple CUDA GPU implementations.

The C code is integrated with Python using ctypes, and the CUDA code is integrated using PyCUDA.

You can learn more about programmatic labeling and Snorkel 
by visiting the [Snorkel website](https://docs.snorkel.ai/docs/25.1/user-guide/intro/welcome-to-snorkel-flow).



## Targeted Labeling Function Types

This repository focuses on accelerating a specific subset of **threshold-based labeling functions** that operate on numerical features. Each labeling function (LF) in this category has the following characteristics:

- **Feature Subset**: Operates on a selected subset of input features (typically 4-20 features per LF)
- **Threshold Values**: Uses predefined threshold values for each feature
- **Comparison Logic**: Employs comparison operators (`>`, `≤`) combined with AND logic
- **Binary Labeling**: Assigns binary labels based on two types:
  - **True-positive LFs**: Return label 1 when all conditions are met
  - **False-positive LFs**: Return label 0 when all conditions are met (i.e., flag suspicious cases)
- **Default Case**: Returns -1 (abstain) when conditions aren't met

### Snorkel labeling functions Example

Example ot type 1 (true positive) and type 2 (false positive) **snorkel** labeling function:

```python
from snorkel.labeling import labeling_function

#Type 1 (true positive)
@labeling_function()
def complex_lf(x):
    # Example with 4 conditions using different operators
    if (x.features[0] < 0.2 and 
        x.features[17] >= 0.5 and 
        x.features[29] > 0.7 and 
        x.features[333] <= 0.9):
        return 1
    return -1

#Type 2 (flase positive)
@labeling_function()
def complex_lf(x):
    # Example with 4 conditions using different operators
    if (x.features[0] < 0.2 and 
        x.features[17] >= 0.5 and 
        x.features[29] > 0.7 and 
        x.features[333] <= 0.9):
        return 0
    return -1
```

## Implementations

The project compares five different implementations:
1. **Native Snorkel** (Python reference implementation)
2. **C multithreading** (CPU-optimized implementation)
3. **CUDA GPU v0 kernel** (Basic GPU implementation)
4. **CUDA GPU v1 kernel** (Optimized GPU implementation)
5. **CUDA GPU v2 kernel** (Further optimized GPU implementation)

## Purpose of benchmark

The benchmark serves to:
- Validate functional equivalence across all implementations
- Verify CUDA kernels produce correct results matching the reference Snorkel implementation
- Test correctness across various input sizes and configurations
- Provide performance comparisons between different approaches

## Test Methodology

The test framework:
1. Generates random test cases with varying parameters:
   - Number of customers
   - Number of features
   - Number of labeling functions
   - Size range of labeling functions
2. For each test case:
   - Computes results using all implementations
   - Validates identical results across implementations
   - Uses assertions to verify result equality


## Installation

```bash
git clone https://github.com/aav-antonov/snorkel-labeling-gpu/
cd snorkel-labeling-gpu
pip install .


```


## Test Run
Validation Test

Verify all implementations produce identical results:

```bash

python test_me.py

```

## Perfomance Benchmark Tests

Run performance comparisons at different scales:

    

```bash

python benchmark_small_size.py

python benchmark_large_size.py
   
python benchmark_extralarge_size.py

```

Each benchmark script will automatically generate its corresponding results file:

    results_small_scale.md

    results_large_scale.md

    results_extralarge_scale.md


## Expected results_small_scale

Hardware:
CPU: AMD Ryzen 5 3600 6-Core Processor
GPU: NVIDIA GeForce RTX 2070

| Method | customer_n | lf_n | features_n | execution_time(s) |
|--------|------------|------|------------|-------------------|
| C_cpucores_12 | 1000 | 1000 | 300 | 0.01690 |
| GPU_kernel_v0 | 1000 | 1000 | 300 | 0.00163 |
| GPU_kernel_v1 | 1000 | 1000 | 300 | 0.00039 |
| GPU_kernel_v2 | 1000 | 1000 | 300 | 0.00065 |
| SnorkelLF | 1000 | 1000 | 300 | 16.61629 |
| C_cpucores_12 | 2000 | 1000 | 300 | 0.02209 |
| GPU_kernel_v0 | 2000 | 1000 | 300 | 0.00170 |
| GPU_kernel_v1 | 2000 | 1000 | 300 | 0.00038 |
| GPU_kernel_v2 | 2000 | 1000 | 300 | 0.00052 |
| SnorkelLF | 2000 | 1000 | 300 | 33.15996 |

## Expected results_large_scale

Hardware:
CPU: AMD Ryzen 5 3600 6-Core Processor
GPU: NVIDIA GeForce RTX 2070

| Method | customer_n | lf_n | features_n | execution_time(s) |
|--------|------------|------|------------|-------------------|
| C_cpucores_12 | 1000000 | 100000 | 400 | 313.22266 |
| GPU_kernel_v0 | 1000000 | 100000 | 400 | 66.37786 |
| GPU_kernel_v1 | 1000000 | 100000 | 400 | 8.09228 |
| GPU_kernel_v2 | 1000000 | 100000 | 400 | 8.36823 |



## Expected results_extralarge_scale

Hardware:
CPU: AMD Ryzen 5 3600 6-Core Processor
GPU: NVIDIA GeForce RTX 2070

| Method | customer_n | lf_n | features_n | execution_time(s) |
|--------|------------|------|------------|-------------------|
| GPU_kernel_v0 | 1000000 | 100000 | 200 | 52.28060 |
| GPU_kernel_v1 | 1000000 | 100000 | 200 | 8.54116 |
| GPU_kernel_v2 | 1000000 | 100000 | 200 | 8.49513 |
| GPU_kernel_v0 | 1000000 | 200000 | 200 | 104.77460 |
| GPU_kernel_v1 | 1000000 | 200000 | 200 | 16.82158 |
| GPU_kernel_v2 | 1000000 | 200000 | 200 | 17.26870 |
| GPU_kernel_v0 | 2000000 | 100000 | 200 | 103.05417 |
| GPU_kernel_v1 | 2000000 | 100000 | 200 | 17.54771 |
| GPU_kernel_v2 | 2000000 | 100000 | 200 | 17.50713 |
| GPU_kernel_v0 | 2000000 | 200000 | 200 | 210.15120 |
| GPU_kernel_v1 | 2000000 | 200000 | 200 | 33.15818 |
| GPU_kernel_v2 | 2000000 | 200000 | 200 | 33.78268 |


