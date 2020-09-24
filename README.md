# Genetic_Classifier
Using Genetic and Simulated Annealing to train Neural networks  


The input is a CSV file with properties (keep this format)
    
    
    Y1 Y2 ...... OUT
    
 0  5   5         0
 
 1  6   7         1
 
 2  55  12        5
 .
 
 .
 
 .
 
 
 ## To Run :

$git clone repo
$cd Genetic_Classifier
$python3 GNN.py
 
## Disciprtion
The code models the nueral network as a set of metrices.
The values in the matrices represent the weights of the nueral network. The feed forward mechanism involves matrix multiplication. The model learns the weights using a genetic algorithm that updates the weights. The hyper parameters and the model structure can be changed in the code. To avoid local(false) minimas a modification is done to the genetic algorithm. If the model detects that it is stagnant it will broden its search else it will focus on a specific region.

This is seen in the following graphs where:
![Alt text](/Graph_1.png?raw=true "Title")
