import numpy as np
import math
import copy

def most_frequent(Lst): 
    counter = 0
    most_freq = Lst[0]   
    for i in Lst: 
        curr_frequency = Lst.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            most_freq = i 
    return most_freq

class label_data:
    def __init__(self,original_data,label,normalized_data):
        self.original_data = original_data
        self.normalized_data = normalized_data
        self.label = label

    def print_original_data_and_label(self):
        print("Label: ",self.label, "original data: ")
        label_data.print_data_item_with_header(self.original_data)
    
    def print_label(self):
        print(self.label)

    @staticmethod
    def print_data_item_with_header(item):
        print("FG: ",item[0], end=' ')
        print("TG: ",item[1], end=' ')
        print("TN: ",item[2], end=' ')
        print("TX: ",item[3], end=' ')
        print("SQ: ",item[4], end=' ')
        print("DR: ",item[5], end=' ')
        print("RH: ",item[6])


class K_means:
    def __init__(self,k=5,tol=0.001, max_iter=300, fname = 'dataset1.csv'):
        self.k = k
        self.tol = tol 
        self.centroids = {}
        self.clusters = {}
        self.clusters_with_names = []
        self.prev_centroids = {}
        self.max_iter = max_iter

        self.fname = fname
        self.data = self.get_data()
        self.labels = self.get_labels()
        self.min_max = self.get_min_max()
        self.normalized_data = self.data
        self.normalize_data()
        self.data_with_info = []

        for i in range(len(self.data)):
            self.data_with_info.append(label_data(self.data[i],self.labels[i],self.normalized_data[i]))

    def get_data(self):
        return np.genfromtxt(self.fname, delimiter=';', usecols=[1,2,3,4,5,6,7], 
            converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

    def get_labels(self):
        dates = np.genfromtxt(self.fname, delimiter=';', usecols=[0])
        labels = []
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

    def get_min_max(self):
        FG = []
        TG = []
        TN = []
        TX = []
        SQ = []
        DR = []
        RH = []
        for item in self.data:
            FG.append(item[0])
            TG.append(item[1])
            TN.append(item[2])
            TX.append(item[3])
            SQ.append(item[4])
            DR.append(item[5])
            RH.append(item[6])
        return [[min(FG),max(FG)], [min(TG),max(TG)], [min(TN),max(TN)], [min(TX),max(TX)], [min(SQ),max(SQ)], [min(DR),max(DR)], [min(RH),max(RH)]]

    def normalize_data(self):
        for i in range(len(self.data)):
            for j in range(len(self.min_max)):
                self.normalized_data[i][j] = (self.data[i][j] - self.min_max[j][0]) / (self.min_max[j][1] - self.min_max[j][0])

    def get_distance(self,data,target):
        return  math.sqrt( pow((data[0]-target[0]),2) + pow((data[1]-target[1]),2) + pow((data[2]-target[2]),2) \
                + pow((data[3]-target[3]),2) + pow((data[4]-target[4]),2) + pow((data[5]-target[5]),2) \
                + pow((data[6]-target[6]),2) )

    def classify(self):
        for i in range(self.k):
            self.centroids[i] = self.data[i]
        
        for i in range(self.max_iter):
            self.clusters = {}

            for i in range(self.k):
                self.clusters[i] =[]
            for data_point in self.data_with_info:
                distances = []
                for centroid in self.centroids:
                    distances.append(self.get_distance(data_point.normalized_data,self.centroids[centroid]))
                cluster = distances.index(min(distances))
                self.clusters[cluster].append(data_point)

                prev_centroids = dict(self.centroids)

                for cluster in self.clusters:
                    pass
                    #self.centroids[cluster] = np.average(self.clusters[cluster],axis=0)
                
                optimized = True

                for c in self.centroids:
                    original_centroid = prev_centroids[c]
                    current_centroid = self.centroids[c]
                    if self.get_distance(current_centroid,original_centroid)*100.0 > self.tol:
                        optimized = False

            if optimized == True:
                print("end reached")
                break
        for cluster in self.clusters:
            most_freq_label = []
            for data_point in self.clusters[cluster]:
                most_freq_label.append(data_point.label)
            cluster_with_name = (most_frequent(most_freq_label), cluster)
            self.clusters_with_names.append(cluster_with_name)


def main():
    k_m = K_means()
    k_m.classify()
    for cluster in k_m.clusters_with_names:
        print(cluster)
   

if __name__ == "__main__":
    main()