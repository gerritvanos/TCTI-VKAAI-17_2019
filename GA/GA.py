import random
import math
import numpy as np
import copy
#uitgebreidere uitleg over keuzes staan in het word document
class GeneticAlgorithm:
    population_size = 150

    genotype_length = 10

    mutation_rate = 0.1

    retain_rate = 0.5

    total_generations = 2000
    # the sum pile, end result for the SUM pile
    # card1 + card2 + card3 + card4 + card5, MUST be 36 
    pile_0_sum = 36
    # the product pile, end result for the PRODUCT pile
    # card1 * card2 * card3 * card4 * card5, MUST be 360 
    pile_1_product = 360

    # the genes array, 30 members, 10 cards each
    population = []
    for i in range(0, population_size):
        population.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    fittnesses =[]
    # used to store random values
    rnd = random.random()

    finished = False
    def run(self):
        # initialise the population (randomly)
        self.init_pop()

        for i in range(self.total_generations):

            self.fittnesses.clear()

            for i in range(len(self.population)):
                self.fittnesses.append(self.fittness(i))
                if self.fittness(i) == 0.0:
                    print(i)
                    print("done")
                    self.display(i)
                    self.finished = True

            if self.finished:
                break
            self.evolve()

    def evolve(self):
        new_population =[]
        old_population = copy.deepcopy(self.population)
        probability = 1/np.array(self.fittnesses)
        probability /= np.sum(probability)
        indexes = np.arange(self.population_size)
        #make children from whole population
        for _ in range(self.population_size - int(self.population_size * self.retain_rate)):
            parent_indexes = np.random.choice(indexes, size=2, replace=False)
            new_population.append(self.crossover(self.population[parent_indexes[0]], self.population[parent_indexes[1]]))

        self.population = new_population

        #retain best part of old population
        retain_indexes = np.argpartition(np.array(self.fittnesses),int(self.population_size * self.retain_rate))
        for i in range(int(self.population_size * self.retain_rate)):
            self.population.append(old_population[retain_indexes[i]])

        #mutate
        indexes = np.arange(10)
        for i in range(self.population_size):
            self.rnd = random.random()
            if self.rnd < self.mutation_rate:
                mutate_indexes = np.random.choice(indexes, size=2, replace=False,)
                self.population[i][mutate_indexes[0]] = self.population[i][mutate_indexes[1]]

    def crossover(self,member1,member2):
        new_member =[1,2,3,4,5,6,7,8,9,10]
        for i in range(self.genotype_length):
            self.rnd = random.random()
            if self.rnd <0.5:
                new_member[i] = member1[i]
            else:
                new_member[i] = member2[i]
        return new_member

    def display(self, n):
        print("with fitness 0.0 the following piles are created")
        print("pile_0 (sum)")
        for i in range(self.genotype_length):
            if self.population[n][i] == 0:
                print(i+1)
        print("pile_1 (product)")
        for  i in range(self.genotype_length):
            if self.population[n][i] == 1:
                print(i+1)


    def fittness(self, n):
        total_sum = 0
        total_prod = 1
        # loop though all genes for this population member
        for i in range(0, self.genotype_length):
            # if the gene value is 0, then put it in the sum (pile 0), and calculate sum
            if self.population[n][i] == 0:
                total_sum += (1 + i)
            # if the gene value is 1, then put it in the product (pile 1), and calculate product
            else:
                total_prod *= 1 + i # +1 because of start at 0 index
        # work out how good this population member is, based on an overall error
        scaled_sum_error = (total_sum - self.pile_0_sum) / self.pile_0_sum
        scaled_prod_error = (total_prod - self.pile_1_product) / self.pile_1_product
        combined_error = math.fabs(scaled_sum_error) + math.fabs(scaled_prod_error)

        return combined_error

    # initialise population
    def init_pop(self):
        for i in range(0, self.population_size):
            for j in range(0, self.genotype_length):
                #randomly create gene values
                self.rnd = random.random()
                if self.rnd < 0.5:
                    self.population[i][j] = 0 #card on pile 0
                else:
                    self.population[i][j] = 1 #card on pile 1

GA = GeneticAlgorithm()
GA.run()