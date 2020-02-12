import numpy as np 
import math
import copy

def get_data(fname):
    return np.genfromtxt(fname, delimiter=';', usecols=[1,2,3,4,5,6,7], 
            converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

def get_labels(fname):
    dates = np.genfromtxt(fname, delimiter=';', usecols=[0])
    labels = []
    if dates[0] > 20001231:
        for label in dates:
            if label < 20010301:
                labels.append('winter')
            elif 20010301 <= label < 20010601:
                labels.append('lente')
            elif 20010601 <= label < 20010901:
                labels.append('zomer')
            elif 20010901 <= label < 20011201:
                labels.append('herfst')
            else: # from 01-12 to end of year
                labels.append('winter')
    else:
        for label in dates:
            if label < 20000301:
                labels.append('winter')
            elif 20000301 <= label < 20000601:
                labels.append('lente')
            elif 20000601 <= label < 20000901:
                labels.append('zomer')
            elif 20000901 <= label < 20001201:
                labels.append('herfst')
            else: # from 01-12 to end of year
                labels.append('winter')
    return labels

def get_min_max(data):
    FG = []
    TG = []
    TN = []
    TX = []
    SQ = []
    DR = []
    RH = []
    for item in data:
        FG.append(item[0])
        TG.append(item[1])
        TN.append(item[2])
        TX.append(item[3])
        SQ.append(item[4])
        DR.append(item[5])
        RH.append(item[6])
    return [[min(FG),max(FG)], [min(TG),max(TG)], [min(TN),max(TN)], [min(TX),max(TX)], [min(SQ),max(SQ)], [min(DR),max(DR)], [min(RH),max(RH)]]

def normalize_data(data,min_max):
    for i in range(len(data)):
        for j in range(len(min_max)):
            data[i][j] = (data[i][j] - min_max[j][0]) / (min_max[j][1] - min_max[j][0])
    return data

def get_distance(data,target):
    return  math.sqrt( pow((data[0]-target[0]),2) + pow((data[1]-target[1]),2) + pow((data[2]-target[2]),2) \
            + pow((data[3]-target[3]),2) + pow((data[4]-target[4]),2) + pow((data[5]-target[5]),2) \
            + pow((data[6]-target[6]),2) )

def sort_by_tuple(lst_of_tup):
    return sorted(lst_of_tup,key=get_key)

def get_key(item):
    return item[1]

def most_frequent(Lst): 
    counter = 0
    most_freq = Lst[0]   
    for i in Lst: 
        curr_frequency = Lst.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            most_freq = i 
    return most_freq

def find_label(target,k):
    differences =[]
    for item in normalized_training_data:
        differences.append(get_distance(item,target))
        
    diff_index =[]
    for i in range(len(differences)):
        diff_index.append((i,differences[i]))
    
    diff_index = sort_by_tuple(diff_index)

    possible_labels =[]
    for i in range(k):
        possible_labels.append(labels[diff_index[i][0]])

    return  most_frequent(possible_labels)

def calculate_efficiency(label_tuple):
    good_bad =[]
    for i in range(len(label_tuple[0])):
        if label_tuple[0][i] == label_tuple[1][i]:
            good_bad.append(True)
        else: 
            good_bad.append(False)
    good_amount = good_bad.count(True)
    return good_amount/len(good_bad) * 100

def procces_training_data():
    training_data_fname = 'dataset1.csv'
    training_data = get_data(training_data_fname)
    global labels 
    labels = get_labels(training_data_fname)
    global min_max
    min_max = get_min_max(training_data)
    global normalized_training_data 
    normalized_training_data = normalize_data(training_data,min_max)

def procces_validation_data(k):
    validation_data_fname = 'validation1.csv'
    validation_data = get_data(validation_data_fname)
    normalized_validation_data = normalize_data(validation_data,min_max)
    validation_labels = get_labels(validation_data_fname)
    retrieved_labels =[]

    for item in normalized_validation_data:
        retrieved_labels.append(find_label(item,k))

    return (retrieved_labels,validation_labels)

def calculate_best_k(start,end):
    eff_dict = {}
    for k in range(start,end+1,1):
        retrieved_labels = procces_validation_data(k)
        efficiency = calculate_efficiency(retrieved_labels)
        eff_dict[k] = efficiency

    best_k = max(eff_dict, key=eff_dict.get)
    return (best_k,eff_dict[best_k] )

def procces_days(k):
    days_fname = 'days.csv'
    days_data = get_data(days_fname)
    normalized_days_data = normalize_data(copy.deepcopy(days_data),min_max)
    for i in range(len(normalized_days_data)):
        found_label = find_label(normalized_days_data[i],k)
        print("found the following label:",found_label)
        print("the data for this label contained:")
        print_data_item_with_header(days_data[i])

def print_data_item_with_header(item):
    print("FG: ",item[0], end=' ')
    print("TG: ",item[1], end=' ')
    print("TN: ",item[2], end=' ')
    print("TX: ",item[3], end=' ')
    print("SQ: ",item[4], end=' ')
    print("DR: ",item[5], end=' ')
    print("RH: ",item[6])

def main():
    k_start = 1
    k_end = 100

    procces_training_data()
    print("calculating best k from " , k_start , " till ", k_end , " based on validation data")
    best_k_info = calculate_best_k(k_start,k_end)
    print("the best k = " , best_k_info[0], " with an efficieny of ", best_k_info[1] , " %")
    print("getting seasons to values in days.csv with the best k value: ",best_k_info[0])
    procces_days(best_k_info[0])

if __name__ == "__main__":
    main()