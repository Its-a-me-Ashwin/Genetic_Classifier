# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:58:11 2019

@author: 91948
"""


import matplotlib.pyplot as plt
import csv
import random
#import cupy as np
import numpy as np
from tqdm import tqdm
import sys





class Neural_Net_Gen:    
    #fuction to normalize data between 0.0 and 1
    #input: numpy vector
    #output: numpy vector
    def normalize(self,data):
        final_result = list()
        for i in data:
            max_ = max(i)
            result = list(map(lambda x: x/abs(max_),i))
            result = np.array(result,dtype = np.float64)
            final_result.append(result)
        final_result = np.array(final_result)
        return final_result
    
    # CSV reader
    #input: file path
    #output: numpy vector
    def read_data (self,path):
        attr = list()
        out = list()
        with open(path) as file:
            data = csv.reader(file)
            for rows in data:
                attr.append(rows[1:-1])
                out.append(rows[-1:][0])
        result = list(zip(attr,out))[1:]
        random.shuffle(result)
        X = list()
        Y = list()
        for x,y in result:
            temp = list(map(lambda e:int (e),x))
            X.append(temp)
            Y.append(int(y))
        X = np.asarray(X)
        X = self.normalize(X)
        Y = np.asarray(Y)
        return (X,Y)
    
    # builds the neural network
    # input: input and output shapes and size of the hidden layers
    # output: builds the modfel with random weights
    def make_matrices(self,input_shape,output_shape,hidden_layers = 1, size = [], sols = 1):
        if len(size) != hidden_layers: return None
        result_mat = list()
        for i in range(sols): # generate population
            matrices = list()
            input_layer_mat = np.random.uniform(low = -1.0, high = 1.0, size = (input_shape,size[0]))
            hidden_layers_mat = list()
            prev = 0
            for hidden_layer in size[1:]:
                hidden_layers_mat.append(np.random.uniform(low = -1.0, high = 1.0, size = (size[prev],hidden_layer)))
                prev += 1
            output_layer = np.random.uniform(low = -1.0, high = 1.0, size = (size[prev],output_shape))
            matrices.append(input_layer_mat)
            for j in hidden_layers_mat:
                matrices.append(j)
            matrices.append(output_layer)
            matrices = np.array(matrices)
            result_mat.append(matrices)
        result_mat = np.array(result_mat)
        return result_mat
    # load data
    
    #converts a given matrix to vector for easy manipulation
    #input: numpy vector
    #output: numpy vector
    def matrix_cvt_vector(self,matrix):
        vector = list()
        for i in range(matrix.shape[0]):
            cur_sol = list()
            for j in range(matrix.shape[1]):
                vec = np.reshape(matrix[i,j],matrix[i,j].size)
                cur_sol.extend(vec)
            vector.append(cur_sol)
        vector = np.array(vector)
        return vector
    
    #converts vector to matrix for easy prediction and use as neuralnetwork
    #input: numpy vector and its requiried shape
    #output: numpy vector
    def vector_to_mat(self,vector_pop_weights, mat_pop_weights):
        mat_weights = []
        for sol_idx in range(mat_pop_weights.shape[0]):
            start = 0
            end = 0
            for layer_idx in range(mat_pop_weights.shape[1]):
                end = end + mat_pop_weights[sol_idx, layer_idx].size
                curr_vector = vector_pop_weights[sol_idx, start:end]
    
                mat_layer_weights = np.reshape(curr_vector, newshape=(mat_pop_weights[sol_idx, layer_idx].shape))
                mat_weights.append(mat_layer_weights)
                start = end
        return np.reshape(mat_weights, newshape=mat_pop_weights.shape) 
    
    #predicts the output for given set of instances
    #input: the generated model, input(X), real output(Y)
    #output: accuracy
    def predict_outputs(self,weights_mat, data_inputs, data_outputs = None,show = False):
        predictions = list()
        for sample_idx in range(data_inputs.shape[0]):
            r1 = data_inputs[sample_idx, :]
            for curr_weights in weights_mat:
                r1 = np.matmul(r1, curr_weights)
                r1 = 1.0/(1.0+np.exp(-1*r1))
            predicted_label = np.where(max(r1) == r1)[0][0]
            #print(predicted_label)np.where(r1 == np.max(r1))[0][0]
            predictions.append(predicted_label)
        
        predictions = np.array(predictions)
        correct_predictions = 0.0
        for i in range(len(predictions)):
            if predictions[i] == data_outputs[i]:
                correct_predictions += 1.0
        accuracy = (correct_predictions/data_outputs.size)*100
        if show:
            #print("Predicted:",predictions)
            print("Acc for entitty:",accuracy)
            return accuracy
        return accuracy
        
    #validates the population to determine which is performing the best
    #input: the generated model, input(X), real output(Y)
    #output: accuracy
    def fitness_pop(self,weights_mat, data_inputs, data_outputs ,show = False):
        accuracy = list()
        for sol_idx in range(weights_mat.shape[0]):
            curr_sol_mat = weights_mat[sol_idx, :]
            accuracy.append(self.predict_outputs(curr_sol_mat, data_inputs, data_outputs,show =  show))
        accuracy = np.array(accuracy)
        if show:
            print("Best acc over the population:",max(accuracy),"tested on",len(data_outputs),"data points")
            return None
        else:
            return accuracy
    
    #select the best n entities in the current population and returns them
    def select_mating_pool(self,pop, fitness, num_parents = 2):
        parents = np.empty((num_parents, pop.shape[1]))
        fitness_sorted = sorted(fitness)[::-1]
        p = 0
        for best_fit in fitness_sorted[:num_parents]:
            idx = np.where(best_fit == fitness)
            #parents[p,:] = pop[idx,:]
        if idx and False:
            return None
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -1.0
            p += 1.0
            if p > num_parents: break
        return parents
    
    
    #combines parents together to get a new entity
    def crossover_1(self,parents, offspring_size,use_rand = False):
        offspring = np.zeros(offspring_size)
        crossover_point = offspring_size[1]/2
        crossover_point = np.uint64(crossover_point)
        for k in range(offspring_size[0]):
            parent1_idx = k%parents.shape[0]
            parent2_idx = (k+1)%parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        if use_rand:
            for k in range(offspring_size[0]):
                for allel in offspring[k]:
                    offspring[k] += random.random()
        return offspring
    
    def mutation_1(self,offspring_crossover, mutation_percent, type_rand = False):
        num_mutations = np.uint32((mutation_percent*offspring_crossover.shape[1])/100)
        mutation_indices = np.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.
            if type_rand:
                offspring_crossover += np.array([random.random() for _ in range(0, offspring_crossover.shape[1]) ]) 
            else:
                random_value = np.random.uniform(-1.0, 1.0, 1)
                offspring_crossover[idx, mutation_indices] = offspring_crossover[idx, mutation_indices] + random_value
        return offspring_crossover
    
    
    def crossover_2(self,parents, offspring_size,use_rand = False):
        offspring = np.zeros(offspring_size)
        take = [random.randint(0,1) for _ in range(int(parents.shape[0]))]
        take = np.array(take)
        crossover_point = offspring_size[1]/2
        crossover_point = np.uint64(crossover_point)
        if not use_rand:
            for k in range(offspring_size[0]):
                parent1_idx = k%parents.shape[0]
                parent2_idx = (k+1)%parents.shape[0]
                for allel in range(len([k])):
                    if take[allel]:
                        offspring[k,allel] = parents[parent1_idx,allel]
                    else:
                        offspring[k,allel] = parents[parent2_idx,allel]
                    offspring[k] += random.random()
        return offspring
    
    
    # The training function
    # takes the data set as input
    def train (self,path,sol_per_pop = 25,
        num_parents_mating = 10,
        epochs = 150,
        max_mutation = 80,
        lowest_mutation = 10,
        cut_off_acc = 95.0,
        stand_off = 50,show = True):
        import math
        import time
        data_inputs,data_outputs = self.read_data(path)
        test_inputs,test_outputs = data_inputs[:int(len(data_inputs)*0.8)],data_outputs[:int(len(data_outputs)*0.8)]
        data_inputs,data_outputs = data_inputs[int(len(data_inputs)*0.8):],data_outputs [int(len(data_outputs)*0.8):]
        
        
        # set up data
        sol_per_pop = 25
        num_parents_mating = 10   
        epochs = 150
        max_mutation = 80
        lowest_mutation = 10
        cut_off_acc = 95.0
        stand_off = 50
        change_per_generation = (max_mutation - lowest_mutation)/epochs
        mutation_percent = max_mutation
        if num_parents_mating > sol_per_pop:
            print("Error in size")
            sys.exit(0)
        
        
        # setup the neuralnetwork  
        pop_weights_mat = self.make_matrices(24,2,2,[150,10],sol_per_pop)
        pop_weights_vector = self.matrix_cvt_vector(pop_weights_mat)
        accuracies = list()
        mutations = list()
        change = 0
        prev = None 
        start_time = time.time()
        for generation in (range(epochs)):
            #print("Generation : ", generation)
            mutation_percent = math.ceil(mutation_percent - generation*change_per_generation)
            if mutation_percent < lowest_mutation:
                mutation_percent = lowest_mutation
            # converting the solutions from being vectors to matrices.
            pop_weights_mat = self.vector_to_mat(pop_weights_vector, 
                                               pop_weights_mat)
        
            # Measuring the fitness of each entity in the population.
            fitness = self.fitness_pop(pop_weights_mat, 
                                  data_inputs, 
                                  data_outputs)
            #print("Fitness")
            if prev == fitness[0]:
                change += 1
            else:
                change = 0
            if change >= stand_off:
                mutation_percent = 90.0
            if change >= 100:
                mutation_percent = random.random()*100.0
                change = 0
            if max(fitness) > cut_off_acc:
                break
            print("Generation",generation,"Best Acc:",round(fitness[0],2),"Mutation Rate:",int(mutation_percent))
            # Selecting the best parents in the population for mating.
            parents = self.select_mating_pool(pop_weights_vector, 
                                            fitness.copy(), 
                                            num_parents_mating)
            offspring_crossover = self.crossover_1(parents,
                                               offspring_size=(sol_per_pop-num_parents_mating, pop_weights_vector.shape[1]))
            offspring_mutation = self.mutation_1(offspring_crossover, mutation_percent=mutation_percent)
            pop_weights_vector[0:parents.shape[0], :] = parents
            pop_weights_vector[parents.shape[0]:, :] = offspring_mutation
            prev = fitness[0]
            accuracies.append(fitness[0])
            mutations.append(mutation_percent)
        if show:
            print("Time taken:",time.time() - start_time)
            plt.plot(accuracies)
            plt.plot(mutations)
            plt.xlabel("Num of Generations")
            plt.ylabel("Blue(Accuracy)/Orange(Mutation Rate)")
            plt.show()
            self.fitness_pop(pop_weights_mat, 
                                  test_inputs, 
                                  test_outputs,True)
        return self.vector_to_mat(pop_weights_vector, pop_weights_mat)
    def fit (self,path,show = False):
        return self.train(path,show = show);
        
# best is the weights of the best generation
GNN = Neural_Net_Gen()
bestPerformingEntity = GNN.fit('Eyes.csv')      
        
