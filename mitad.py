import sys
import time
import random as rnd
import numpy as np
from matplotlib import pyplot as plt
import MultiNEAT as NEAT
from MultiNEAT.viz import plot_nn

def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    error = 0
    maxval = 100
    for i in range(maxval):
        net.Flush()
        net.Input(np.array([float(i), 1]))
        for _ in range(2):
            net.Activate()
        o = net.Output()
        error += abs(i/2 -o[0])
    
    fitness = -error/maxval
    return fitness

################################################
params = NEAT.Parameters()
params.PopulationSize = 100
params.OverallMutationRate = 1.0

params.WeightMutationMaxPower = 0.5
params.WeightReplacementMaxPower = 8
params.WeightMutationRate = 0.25
params.WeightReplacementRate = 0.5

params.MutateAddNeuronProb = 0.05
params.MutateAddLinkProb = 0.3
params.MutateRemLinkProb = 0.01
params.MutateWeightsProb = 0.05
params.MutateRemSimpleNeuronProb = 0.05

params.MinActivationA = 4.9
params.MaxActivationA = 4.9

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_Linear_Prob = 1.0

################################################

def evolve():
    g = NEAT.Genome(0, 2, 0, 1, False, NEAT.ActivationFunction.LINEAR,
                    NEAT.ActivationFunction.LINEAR, 0, params, 0)
    pop = NEAT.Population(g, params, True, 1.0, 1)
    pop.RNG.Seed(int(time.clock()*100))

    generations = 0
    for generation in range(50):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = []
        for genome in genome_list:
            fitness_list.append(evaluate(genome))
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()
        generations = generation
        best = -max(fitness_list)
        bestG = pop.GetBestGenome()
        
        plot_nn(bestG)
        plt.pause(0.01)
        plt.ion()
        plt.show(block=False)
        
        print("Mejor fitness [",generation,"]: ",best)
        if best < 0.01:
            break

    testNet = NEAT.NeuralNetwork()
    bestG.BuildPhenotype(testNet)
    for i in range(10):
        testNet.Flush()
        testNet.Input(np.array([float(100+2*i), 1]))
        for _ in range(2):
            testNet.Activate()
        o = testNet.Output()
        print(100+2*i,"/ 2 = ",o[0]) 
        
    return generations


gen = evolve()
input("Presiona ENTER para finalizar")
