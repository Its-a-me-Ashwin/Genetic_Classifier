# Genetic_Classifier
Using Genetic and Simulated Annealing to train Neural networks  


The input is a CSV file with properties
    Y1 Y2 ...... OUT
 0  5   5         0
 1  6   7         1
 2  55  12        5
 .
 .
 .
 
 The code models the nueral network as a set of metrices (structure can be changed if needed).
 The weights of the matrices are updated using a getetic algortihm. The traing also uses simulated aneeling to overcome local     minimas.
 
In the graphs the Blue curve is the accuracy and orange is the current mutation rate.
If the accuracy plataues the mutation rate is raised to try and find a better minima.
