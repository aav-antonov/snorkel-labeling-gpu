
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <vector>



typedef struct {
    int start;
    int end;
    int lf_number;
    int *lf_outcome;
    int *lf_features_shape;
    int *lf_features;
    int *lf_features_compare;
    float *lf_features_thresholds;
    
    float *customers_features;
    int features_number;
    int customer_number;
    
    int *results;

} ThreadData;

void* label_customers(void* arg) {
    ThreadData *data = (ThreadData*) arg;
    
    //change TID = 1 to get printout
    int TID = -1;
    for (int tid = data->start; tid < data->end; tid++) {

        if (tid < data->customer_number) {

            float *customers_features_tid = &data->customers_features[data->features_number * tid];
            int *results_tid = &data->results[2 * tid];

            if (tid == TID) {
                printf("customer_number %d\n", data->customer_number);
                printf("lf_number %d\n", data->lf_number);
                printf("features_number %d\n", data->features_number);
            }
            
            int feature_start = 0;
            for (int i = 0; i < data->lf_number; i++) {

                int rule_ok = 1;
                for (int f = feature_start; f < feature_start + data->lf_features_shape[i]; f++) {
                    int f_id = data->lf_features[f];
                    float threshold = data->lf_features_thresholds[f];
                    int compare = data->lf_features_compare[f];
                   
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
                
                //logic for lf_outcome:
                if(rule_ok == 1){
                    if(data->lf_outcome[i] == 1){
                        results_tid[0]++;
                    }else{
                        results_tid[1]++;
                    }
                }

                // move to next lf
                feature_start += data->lf_features_shape[i];
            }
        }
     }
    pthread_exit(NULL);
}

extern "C" {

void label_customers_interface(
int lf_number,
int *lf_outcome,
int *lf_features_shape,
int *lf_features,
int *lf_features_compare,
float  *lf_features_thresholds,

float  *customers_features,
int features_number,
int customer_number,
    
int num_threads,
    
int *results

    
) {

    

    int min_customer_per_thread = 1000;

    if (customer_number < num_threads * min_customer_per_thread ){
       num_threads = (int) customer_number / min_customer_per_thread +1;
    }


    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int customers_per_thread = (int)(customer_number / num_threads) +1;

    for (int t = 0; t < num_threads; t++) {
        
                
        thread_data[t].start = t * customers_per_thread;
        thread_data[t].end = thread_data[t].start + customers_per_thread;
        
        thread_data[t].lf_number = lf_number;
        thread_data[t].lf_outcome = lf_outcome;
        thread_data[t].lf_features_shape = lf_features_shape;
        thread_data[t].lf_features = lf_features;
        thread_data[t].lf_features_compare = lf_features_compare;
        thread_data[t].lf_features_thresholds = lf_features_thresholds;
        
        thread_data[t].customers_features = customers_features;
        thread_data[t].features_number = features_number;
        thread_data[t].customer_number = customer_number;
        
        thread_data[t].results = results;

        int rc = pthread_create(&threads[t], NULL, label_customers, (void*)&thread_data[t]);
        if (rc) {
            printf("Error: unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

}//extern "C"
