#include <stdio.h>


#define block_size 32


__global__ void reshape_lf_features_kernel(
    int lf_number,
    int *lf_features_shape,        // Number of features for each LF
    int *lf_features_start,        // Starting index in flattened array for each LF
    int *lf_features,              // Flattened array of all feature IDs
    int *lf_features_compare,      // Flattened array of all comparison signs
    float *lf_features_thresholds, // Flattened array of all thresholds,

    // Output arrays
    int *lf_features_2d,
    int *lf_features_compare_2d,
    float *lf_features_thresholds_2d
) {
    int lf_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (lf_idx < lf_number) {
        int lf_features_shape_idx = lf_features_shape[lf_idx];
        int lf_features_start_idx = lf_features_start[lf_idx];

        for (int j = 0; j < lf_features_shape_idx; ++j) {

            int output_idx = j * lf_number + lf_idx; // This implements your column-major logic
            int flat_idx = lf_features_start_idx + j;

            lf_features_2d[output_idx] = lf_features[flat_idx];
            lf_features_compare_2d[output_idx] = lf_features_compare[flat_idx];
            lf_features_thresholds_2d[output_idx] = lf_features_thresholds[flat_idx];

        }

    }
}

template<int features_number_MAX>
__global__ void comp_v2_label_customers(
    int lf_number,
    int *lf_outcome,
    int *lf_features_shape,

    int *lf_features_2d,
    int *lf_features_compare_2d,
    float *lf_features_thresholds_2d,

    float *customers_features,
    int features_number,
    int customer_number,

    int *results

) {

    __shared__ float shared_features[features_number_MAX];
    __shared__ int shared_results[2];

    if (threadIdx.x == 0) {
        shared_results[0] = 0;
        shared_results[1] = 0;
    }

    int customer_id = blockIdx.x;

    if (customer_id >= customer_number) return;

    // Load customer features into shared memory
    float *customer_features_global = &customers_features[features_number * customer_id];
    for (int i = threadIdx.x; i < features_number; i += blockDim.x) {
        shared_features[i] = customer_features_global[i];
    }
    __syncthreads();

    // Each thread processes a subset of labeling functions
    for (int lf_idx = threadIdx.x; lf_idx < lf_number; lf_idx += blockDim.x) {
        int rule_ok = 1;
        int lf_shape = lf_features_shape[lf_idx];

        // Check all features for this labeling function
        for (int f = 0; f < lf_shape; f++) {

            int output_idx = f * lf_number + lf_idx;

            int f_id = lf_features_2d[output_idx];
            float threshold = lf_features_thresholds_2d[output_idx];
            int compare = lf_features_compare_2d[output_idx];

            if (compare == 1) {
                if (shared_features[f_id] > threshold) {
                    rule_ok = 0;
                    break;
                }
            } else {
                if (shared_features[f_id] <= threshold) {
                    rule_ok = 0;
                    break;
                }
            }

        }

        // Apply labeling function outcome if rule passed
        if (rule_ok == 1) {
            if (lf_outcome[lf_idx] == 1) {
                atomicAdd(&shared_results[0], 1);  // Atomic for positive count
            } else {
                atomicAdd(&shared_results[1], 1);  // Atomic for negative count
            }
        }

    }
    __syncthreads();

    if (threadIdx.x == 0) {
        results[2 * customer_id + 0] = shared_results[0];
        results[2 * customer_id + 1] = shared_results[1];
    }
}

template __global__ void comp_v2_label_customers<400>(int, int*, int*, int*, int*, float*, float*, int, int, int*);

//---------------------------------------------------------------------//



template<int features_number_MAX>
__global__ void comp_v1_label_customers(
    int lf_number,
    int *lf_outcome,
    int *lf_features_shape,
    int *lf_features_start,  // Added input for feature start positions
    int *lf_features,
    int *lf_features_compare,
    float *lf_features_thresholds,

    float *customers_features,
    int features_number,
    int customer_number,

    int *results

) {

    __shared__ float shared_features[features_number_MAX];
    __shared__ int shared_results[2];

    if (threadIdx.x == 0) {
        shared_results[0] = 0;
        shared_results[1] = 0;
    }

    int customer_id = blockIdx.x;

    if (customer_id >= customer_number) return;

    // Load customer features into shared memory
    float *customer_features_global = &customers_features[features_number * customer_id];
    for (int i = threadIdx.x; i < features_number; i += blockDim.x) {
        shared_features[i] = customer_features_global[i];
    }
    __syncthreads();

    // Each thread processes a subset of labeling functions
    for (int i = threadIdx.x; i < lf_number; i += blockDim.x) {
        int rule_ok = 1;
        int feature_start = lf_features_start[i];  // Directly use precomputed start position
        int feature_end = feature_start + lf_features_shape[i];

        // Check all features for this labeling function
        for (int f = feature_start; f < feature_end; f++) {
            int f_id = lf_features[f];
            float threshold = lf_features_thresholds[f];
            int compare = lf_features_compare[f];

            if (compare == 1) {
                if (shared_features[f_id] > threshold) {
                    rule_ok = 0;
                    break;
                }
            } else {
                if (shared_features[f_id] <= threshold) {
                    rule_ok = 0;
                    break;
                }
            }

        }

        // Apply labeling function outcome if rule passed
        if (rule_ok == 1) {
            if (lf_outcome[i] == 1) {
                atomicAdd(&shared_results[0], 1);  // Atomic for positive count
            } else {
                atomicAdd(&shared_results[1], 1);  // Atomic for negative count
            }
        }

    }
    __syncthreads();

    if (threadIdx.x == 0) {
        results[2 * customer_id + 0] = shared_results[0];
        results[2 * customer_id + 1] = shared_results[1];
    }
}


template __global__ void comp_v1_label_customers<400>(int, int*, int*, int*, int*, int*, float*, float*, int, int, int*);


//-------------- Zero level optimised version 0 -------------------//

__global__ void comp_v0_label_customers(
    int lf_number,
    int *lf_outcome,
    int *lf_features_shape,
    int *lf_features,
    int *lf_features_compare,
    float *lf_features_thresholds,

    float *customers_features,
    int features_number,
    int customer_number,

    int *results

) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < customer_number) {
        float *customers_features_tid = &customers_features[features_number * tid];
        int *results_tid = &results[2 * tid];

        int count0 = 0, count1 = 0;

        int feature_start = 0;
        for (int i = 0; i < lf_number; i++) {
            int rule_ok = 1;

            // Check all features for this labeling function
            for (int f = feature_start; f < feature_start + lf_features_shape[i]; f++) {
                int f_id = lf_features[f];
                float threshold = lf_features_thresholds[f];
                int compare = lf_features_compare[f];

                if (compare == 1) {
                    if (customers_features_tid[f_id] > threshold) {
                        rule_ok = 0;
                        break;
                    }
                } else {
                    if (customers_features_tid[f_id] <= threshold) {
                        rule_ok = 0;
                        break;
                    }
                }
            }

            // Apply labeling function outcome if rule passed
            //logic for lf_outcome:
            if(rule_ok == 1){
               if(lf_outcome[i] == 1){
                  count0++;

               }else{
                  count1++;
               }
            }

            // Move to next labeling function's features
            feature_start += lf_features_shape[i];
        }

        results_tid[0]=count0;
        results_tid[1]=count1;
    }
}