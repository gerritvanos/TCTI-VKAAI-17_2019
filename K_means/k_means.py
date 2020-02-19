import numpy as np
import math
import matplotlib.pyplot as plot
import copy
import random

def most_frequent(Lst): 
    counter = 0
    most_freq = Lst[0]   
    for i in Lst: 
        curr_frequency = Lst.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            most_freq = i 
    return most_freq

def inList(array, lst):
    for element in lst:
        if np.array_equal(element, array):
            return True
    return False

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
    def __init__(self,k=3,tol=0.001, max_iter=30000, itter_per_k = 10, fname = 'dataset1.csv'):
        self.k = k
        self.tol = tol 
        self.optimized = False
        self.centroids = {}
        self.clusters = {}
        self.clusters_with_names = []
        self.prev_centroids = {}
        self.max_iter = max_iter
        self.itter = itter_per_k
        self.start_points = []
        self.total_distances = []
        self.best_start_points = {}
        self.random_points = []

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

    def set_start_centroids(self):
        self.centroids = {}
        self.random_points = list()
        for i in range(self.k):
            #generate random points until a point is found that is not used yet
            random_point = self.data[random.randint(0,len(self.data)-1)]
            while inList(random_point,self.random_points):
                random_point = self.data[random.randint(0,len(self.data)-1)]
            self.random_points.append(random_point)

            self.centroids[i] = random_point #set found point as centroid
        self.start_points.append(self.centroids) # save centroids

    def find_best_start_point(self):
        self.start_points = []
        self.total_distances = []
        for i in range(self.itter):
            self.set_start_centroids()
            self.cluster()
            self.total_distances.append(self.calculate_total_centroid_distance_for_all_clusters(self.clusters))
        self.best_start_points = self.start_points[self.total_distances.index(min(self.total_distances))]
        self.centroids = self.best_start_points
        return self.best_start_points


    def cluster(self):
        for i in range(self.max_iter):
            self.clusters = {} # clear clusters
            prev_centroids = copy.deepcopy(self.centroids)
            
            for k in range(self.k):
                self.clusters[k] =[] #create k empty klusters 
            #add data point to cluster where centroid is closest     
            for data_point in self.data_with_info:
                distances = []
                for centroid in self.centroids:
                    distances.append(self.get_distance(data_point.normalized_data,self.centroids[centroid]))
                cluster = distances.index(min(distances))
                self.clusters[cluster].append(data_point)

            #move centroid to middle of cluster 
            for cluster in self.clusters:
                data_array = []
                for i in range(len(self.clusters[cluster])):
                    data_array.append(self.clusters[cluster][i].normalized_data)
                if data_array != []:
                    self.centroids[cluster] = self.calculate_average_from_2d_array(data_array)
            
            #check if centroids moved
            self.optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if self.get_distance(current_centroid,original_centroid)*100.0 > self.tol:
                    self.optimized = False

            #if centroids not moved break
            if self.optimized == True:
                break
    
    def vote_names(self):     
        self.clusters_with_names = []       
        for cluster in self.clusters:
            most_freq_label = []
            for data_point in self.clusters[cluster]:
                most_freq_label.append(data_point.label)
            cluster_with_name = (most_frequent(most_freq_label), cluster)
            self.clusters_with_names.append(cluster_with_name)

    def calculate_total_centroid_distance_for_all_clusters(self,clusters):
        total =0
        for cluster in clusters:
            total += pow(self.calculate_total_centroid_distance(self.clusters[cluster],self.centroids[cluster]),2)
        return total

    def calculate_total_centroid_distance(self,cluster,centroid):
        total_centroid_distance = 0
        for data_point in cluster:
            total_centroid_distance += pow(self.get_distance(centroid,data_point.normalized_data),2)
        return total_centroid_distance

    def elbow(self,max_k):
        centroid_distances =[]
        k_s =[]
        for k in range(2,max_k+1):
            self.k = k
            self.find_best_start_point()
            self.cluster()
            k_s.append(k)
            centroid_distances.append(self.calculate_total_centroid_distance_for_all_clusters(self.clusters))

        plot.plot(k_s,centroid_distances, marker='o', linewidth = 2)
        plot.ylabel("total centroid distance")
        plot.xlabel("K")
        plot.show()
            
def main():
    k_m = K_means()
    k_m.find_best_start_point()
    k_m.cluster()
    k_m.vote_names()
    print("best k = 3 based on elbow method as the graph will show. \nThe 3 clusters with their voted name will be printed below" )
    print("at this point the differential is not that high anymore also the cluster names are what i excpect")
    for cluster in k_m.clusters_with_names:
        print(cluster)
    k_m.elbow(15)

   

if __name__ == "__main__":
    main()