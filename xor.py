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

    net.Flush()
    net.Input(np.array([1., 0., 1.]))
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(1 - o[0])

    net.Flush()
    net.Input([0, 1, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(1 - o[0])

    net.Flush()
    net.Input([1, 1, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(o[0])

    net.Flush()
    net.Input([0, 0, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    error += abs(o[0])
    
    fitness = (4 - error) ** 2
    return fitness

################################################
params = NEAT.Parameters()
params.PopulationSize = 100
params.OverallMutationRate = 1.0

params.WeightMutationMaxPower = 0.5
params.WeightReplacementMaxPower = 8
params.WeightMutationRate = 0.25
params.WeightReplacementRate = 0.5

params.MutateAddNeuronProb = 0.001
params.MutateAddLinkProb = 0.3
params.MutateRemLinkProb = 0.0
params.MutateWeightsProb = 0.05

params.MinActivationA = 4.9
params.MaxActivationA = 4.9

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0
params.ActivationFunction_UnsignedStep_Prob = 0.0

################################################

def evolve():
    g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    pop = NEAT.Population(g, params, True, 1.0, 1)
    pop.RNG.Seed(int(time.clock()*100))

    generations = 0
    for generation in range(1000):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = []
        for genome in genome_list:
            fitness_list.append(evaluate(genome))
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()
        generations = generation
        best = max(fitness_list)
        bestG = pop.GetBestGenome()
        plot_nn(bestG)
        plt.pause(0.001)
        plt.ion()
        plt.show(block=False)
        print("Mejor fitness [",generation,"]: ",best)
        if best > 15.9:
            break

    return generations


gen = evolve()
print('Generaciones para resolver el XOR:', gen)
input("Presiona ENTER para finalizar")
