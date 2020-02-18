import numpy as np
import math
import matplotlib.pyplot as plot
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
    def __init__(self,k=3,tol=0.00000000, max_iter=30000, fname = 'dataset1.csv'):
        self.k = k
        self.tol = tol 
        self.optimized = False
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

    def calculate_average_from_2d_array(self,array):
        sums =[]
        average = []
        for i in range(len(array[0])):
            sums.append(0)
            average.append(0)
        for i in range(len(array)):
            for j in range(len(array[i])):
                sums[j] += array[i][j]
        for i in range(len(sums)):
            average[i] = sums[i]/len(array)
        return np.array(average)


    def cluster(self):
        self.centroids = {}
        self.clusters_with_names = []
        for i in range(self.k):
            self.centroids[i] = self.data[i]
        
        for i in range(self.max_iter):
            self.clusters = {}
            prev_centroids = copy.deepcopy(self.centroids)

            for k in range(self.k):
                self.clusters[k] =[]
            for data_point in self.data_with_info:
                distances = []
                for centroid in self.centroids:
                    distances.append(self.get_distance(data_point.normalized_data,self.centroids[centroid]))
                cluster = distances.index(min(distances))
                self.clusters[cluster].append(data_point)
   
            for cluster in self.clusters:
                data_array = []
                for i in range(len(self.clusters[cluster])):
                    data_array.append(self.clusters[cluster][i].normalized_data)
                self.centroids[cluster] = self.calculate_average_from_2d_array(data_array)
            

            if i >= 1:
                self.optimized = True
                for c in self.centroids:
                    original_centroid = prev_centroids[c]
                    current_centroid = self.centroids[c]
                    if self.get_distance(current_centroid,original_centroid)*100.0 > self.tol:
                        self.optimized = False
            
            if self.optimized == True:
                print("bij k= ", self.k , " aantal iteraties: ",i)
                break
            
        for cluster in self.clusters:
            most_freq_label = []
            for data_point in self.clusters[cluster]:
                most_freq_label.append(data_point.label)
            cluster_with_name = (most_frequent(most_freq_label), cluster)
            self.clusters_with_names.append(cluster_with_name)

    def elbow(self,max_k):
        centroid_distances =[]
        k_s =[]
        for k in range(2,max_k+1):
            self.k = k
            self.cluster()
            k_s.append(k)
            total_cluster_centroid_distances = []
            for cluster in self.clusters:
                for data_point in self.clusters[cluster]:
                    total_cluster_centroid_distances.append(self.get_distance(data_point.normalized_data,self.centroids[cluster]))
                total_cluster_centroid_distance = pow(np.mean(total_cluster_centroid_distances),2)
            centroid_distances.append(total_cluster_centroid_distance)


        plot.plot(k_s,centroid_distances)
        plot.ylabel("total centroid distance")
        plot.xlabel("K")
        plot.show()
            
def main():
    k_m = K_means()
    k_m.cluster()
    print("best k = 3 based on elbow method as the graph will show. \nThe 3 clusters with their voted name will be printed below")
    for cluster in k_m.clusters_with_names:
        print(cluster)
    k_m.elbow(10)

   

if __name__ == "__main__":
    main()