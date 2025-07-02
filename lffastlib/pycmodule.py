import ctypes
import numpy as np
import os

from lffastlib.pycompile import compile_c
from lffastlib.pycutils import LabelingFunctions, CustomerFeatures


class ComputeLabelingFunctions:
    
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
        
        # compile c_code
        current_directory = os.path.dirname(os.path.abspath(__file__))
        files_c = [ 'c_code/comp_cust.c']
        self.lib_c = compile_c(current_directory,  files_c )

        self.lf_number = self.LF.lf_number
        self.features_number = self.CF.features_number
        self.customer_number = self.CF.customer_number

        # Convert data to ctypes
        self.lf_features_shape_ctypes = self.LF.lf_features_shape.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))# 
        self.lf_features_ctypes = self.LF.lf_features.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.lf_features_compare_ctypes = self.LF.lf_features_compare.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.lf_outcome_ctypes = self.LF.lf_outcome.astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self.lf_features_thresholds_ctypes = np.ascontiguousarray(self.LF.lf_features_thresholds, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.customers_features_ctypes = np.ascontiguousarray(self.CF.customers_features, dtype=np.float32).flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))



    def compute(self, threads = 8):

        # results.shape = (customer_number, 2)
        #       where for each  customer:
        #       results[0] - counts lf_type1 == true
        #       results[1] - counts lf_type0 == true


        results = np.zeros((self.customer_number, 2), dtype=np.int32)
        results_ctypes = results.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # Call the C function
        self.lib_c.label_customers_interface(
            self.lf_number,
            self.lf_outcome_ctypes,
            self.lf_features_shape_ctypes,
            self.lf_features_ctypes,
            self.lf_features_compare_ctypes,
            self.lf_features_thresholds_ctypes,
            
            self.customers_features_ctypes,
            self.features_number,
            self.customer_number,
            threads,
            
            results_ctypes,
        )

        return results.flatten()
    


    




