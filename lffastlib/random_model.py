
from random import sample, shuffle
import random
import pandas as pd
import json

def generate_customer_list(n, k):
    cust_A1 = [f'cust_A1_{i+1}' for i in range(n)]
    cust_B0 = [f'cust_B0_{i+1}' for i in range(k)]
    return cust_A1, cust_B0



def select_and_mix_samples(listA, listB, rateA=0.8, rateB=0.1):
    n_A = int(len(listA) * rateA)
    n_B = int(len(listB) * rateB)
    
    subsetA = sample(listA, n_A)
    subsetB = sample(listB, n_B)
    
    setA = set(subsetA)
    setB = set(subsetB)
    
    exclusiveListB = [x for x in listB if x not in setB]
    exclusiveListA = [x for x in listA if x not in setA]
    listTop = subsetA + subsetB
    listDown = exclusiveListB + exclusiveListA
    
    assert not set(listTop).intersection(set(listDown))
    
    shuffle(listTop)
    shuffle(listDown)
    return listTop + listDown

def generate_feature(n, k):
    values = [random.uniform(0, 100) for _ in range(n + k)]
    values.sort()
    return values

def zip_and_sort_by_listA(listV, listA):
    sorted_pairs = sorted(zip(listV, listA), key=lambda x: x[1])
    sorted_listV, sorted_listA = zip(*sorted_pairs)
    return list(sorted_listV), list(sorted_listA)

def calculate_accuracy(df, threshold, sign, feature_id, lf_type):
    # Assuming the actual labels are in column `(0, 1)`
    actual_labels = df['label']

    # Predict labels based on the threshold T
    if sign == '>':
        predicted_labels = df[feature_id] > threshold
    else:
        predicted_labels = df[feature_id] <= threshold
    
    predicted_labels = predicted_labels.astype(int)  # Convert boolean to integer (0 or 1)
    
    if lf_type == 0:
        predicted_labels = 1 - predicted_labels
        
    

    # Calculate accuracy
    accuracy = (predicted_labels == actual_labels).mean()
    print("f_accuracy", lf_type, sign, accuracy)
    #return accuracy



def gen_features(lf_type, f_size, cust_A1, cust_B0):
    
    
    feature_val = []
    feature_lf, sign_lf, threshold_lf = [], [], []
    for i in range(f_size):
        
        listA, listB = cust_A1, cust_B0
        if lf_type == 0:
            listA, listB =  cust_B0, cust_A1
        
        ## define bias for the fearture
        rateA =  random.uniform(0.5, 0.65)
        rateB =  random.uniform(0.35, 0.5)
        
        sign = random.choice(['>', '<='])
        
        if sign == '>':
            listA, listB = listB, listA
        
        listM = select_and_mix_samples(listA, listB, rateA = rateA, rateB = rateB)
        listV = generate_feature(len(listA), len(listB))
        
        index_th = int(len(listA) * rateA + len(listB) * rateB)
        threshold = listV[index_th]
        
    
        listV, listM  = zip_and_sort_by_listA(listV, listM)
        feature_id = f'f{i}_t{lf_type}'

        
        feature_val.append(listV)
        feature_lf.append(feature_id)
        sign_lf.append(sign)
        threshold_lf.append(threshold)
        
    return feature_val, threshold_lf, sign_lf, feature_lf
    
def generate_lf(count_lf, lf_json, lf_type, n, threshold_lf, sign_lf, feature_lf):    
    
    
    for i in range(n):
        lf_size = random.randint(4, 10)
        lf_id = f"lf{lf_type}_Q{i}"
        
        # Get the indices of the elements to sample
        indices = random.sample(range(len(threshold_lf)), lf_size)
    
        # Create the sampled lists
        threshold_lf_i = [threshold_lf[i] for i in indices]
        sign_lf_i = [sign_lf[i] for i in indices]
        feature_lf_i = [feature_lf[i] for i in indices]
        
        lf_param = {}
        lf_param['id'] = lf_id
        lf_param['class'] = lf_type
        lf_param['feature'] = feature_lf_i 
        lf_param['sign'] = sign_lf_i
        lf_param['threshold'] = threshold_lf_i
    
        lf_json[count_lf] = lf_param
        count_lf +=1
        
    return lf_json, count_lf    


class GenerateBiasRandomSamples:
    
    def __init__(self, customers = [1000, 1000], features_number = 100, lf_number = 500):
    
        [n, k] = customers
        cust_A1, cust_B0 = generate_customer_list(n, k)
        list_cust = cust_A1 + cust_B0
        list_cust = sorted(list_cust)
        labels = [1] * n + [0] * k 
    
    
        ## Generate Features: df
        df = pd.DataFrame({'id': list_cust , 'label': labels})
        
        print(f"Generating {features_number} type 1 features, customer type 1 (SAR): {n}, customer type 0: {k}")
        
        ## features for lf type = 1
        lf_type, f_size = 1, features_number
        feature_val1, threshold_lf1, sign_lf1, feature_lf1 = gen_features(lf_type, f_size, cust_A1, cust_B0)
    
        df1 = pd.DataFrame(feature_val1).transpose()
        df1.columns = feature_lf1
        df = pd.concat([df, df1], axis=1)
        
        print(f"Generating {features_number} type 0 features, customer type 1 (SAR): {n}, customer type 0: {k}")
        
        ## features for lf type = 0
        lf_type, f_size = 0, features_number
        feature_val0, threshold_lf0, sign_lf0, feature_lf0 = gen_features(lf_type, f_size, cust_A1, cust_B0)
    
        df0 = pd.DataFrame(feature_val0).transpose()
        df0.columns = feature_lf0
        self.df = pd.concat([df, df0], axis=1)
    
        
        ## Generate Labelling Functions (as randomly selected set of featurs)
    
        count_lf = 0
        self.lf_json = {}
        
        print(f"Generate {lf_number} Labelling Functions of type 1")
        ## generatye lf type = 1
        lf_type, n = 1, lf_number
        self.lf_json, count_lf =  generate_lf(count_lf, self.lf_json, lf_type, n, threshold_lf1, sign_lf1, feature_lf1)
        
        print(f"Generate {lf_number} Labelling Functions of type 0")
        ## generatye lf type = 0
        lf_type, n = 0, lf_number
        self.lf_json, count_lf =  generate_lf(count_lf, self.lf_json, lf_type, n, threshold_lf0, sign_lf0, feature_lf0)
    
    def save(self, filename):
        
        # Save the best (optimalthresholds) config 
        with open(f"{filename}_opt.json", 'w') as f:
            json.dump(self.lf_json, f, indent=4)
        self.df.to_csv(f'{filename}.csv', index=False)



        # Update the 'threshold' random values
        for key in self.lf_json:
            self.lf_json[key]['threshold'] = [random.uniform(5, 10) for _ in self.lf_json[key]['threshold']]

        # Save the updated dictionary back to a JSON file
        with open(f"{filename}_none_opt.json", 'w') as file:
            json.dump(self.lf_json, file, indent=4)
    
    
    
    