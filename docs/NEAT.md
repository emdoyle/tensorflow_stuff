---
layout: notebook
---

# Using NEAT to Evolve Neural Networks


```python
import MultiNEAT as NEAT
```

This was sort of a pain to install, but the good news is it's all set up now and I have openCV 3.3.0 installed as well.  Basically [this](http://multineat.com/docs.html) is the extent of the documentation since I don't believe this library has been widely used.  It's also not fully compatible with the current `boost` version (1.65) because one of the header files was removed (`numeric.hpp`) so I used `cython` to build it instead (although the new files related to Traits require me to have `boost` installed anyway.

### What is NEAT?

NEAT stands for ["NeuroEvolution of Augmenting Topologies"](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) and is a genetic algorithm for evolving neural networks.  It is based on the idea that mutation, speciation, and natural selection can begin with very simple structures and eventually evolve complex topologies that maximize a given fitness function.

There are really two distinct ways to use NEAT:

    1. Decide on the topology of the neural network ahead of time and simply evolve connection
    weights and biases.

![png](assets/PreDetermTopo.png)

    2. Start with a very simple network consisting of only input and output layers and evolve new nodes
    and connections along with connection weights and biases.

![png](assets/EvolvedTopo.png)

### Evolving a Neural Network That Outputs [1,0]

To get started with NEAT, I am going to go through a toy example.  It will involve initializing a population of simple genomes, constructing neural network phenotypes based on these genomes, defining an evaluation function to determine the fitness of a given phenotype, and defining an epoch cycle to iterate through new generations. (NOTE: This code will primarily come from the [documentation page](http://multineat.com/docs.html))


```python
params = NEAT.Parameters()

params.PopulationSize = 100

# The Genome constructor (in C++) is copied below
# Genome::Genome(unsigned int a_ID,
#                    unsigned int a_NumInputs,
#                    unsigned int a_NumHidden, // ignored for seed type == 0, specifies number of hidden units if seed type == 1
#                    unsigned int a_NumOutputs,
#                    bool a_FS_NEAT, ActivationFunction a_OutputActType,
#                    ActivationFunction a_HiddenActType,
#                    unsigned int a_SeedType,
#                    const Parameters &a_Parameters)

genome = NEAT.Genome(0, 3, 0, 2, False,
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                     NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)  
```

The author of the documentation (likely also the author of the library) tells us to: "Always add one extra input. The last input is always used as bias and also when you activate the network always set the last input to 1.0 (or any other constant non-zero value)." Knowing this, the network we defined just now has 2 'real' inputs and 2 outputs.


```python
# The Population constructor (again in C++) is copied below
# Population::Population(const Genome& a_Seed, const Parameters& a_Parameters,
#     bool a_RandomizeWeights, double a_RandomizationRange, int a_RNG_seed)
pop = NEAT.Population(genome, params, True, 1.0, 0)
```

Now we have a population of 100 simple neural network genomes with randomized weights.  Next step is to define an evaluation function, which will require us to build phenotypes from these genomes.


```python
def evaluate(genome):

    # this creates a neural network (phenotype) from the genome
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    # let's input just one pattern to the net, activate it once and get the output
    net.Input( [ 1.0, 0.0, 1.0 ] )
    net.Activate()
    output = net.Output() 

    # the output can be used as any other Python iterable.
    # instead of the original decision to select for '0' output on the first output,
    # I will select for the output [1,0]
    fitness = output[0] + (1 - output[1])
    return fitness
```


```python
for generation in range(10): # run for 10 generations

    # retrieve a list of all genomes in the population
    genome_list = NEAT.GetGenomeList(pop)

    fitness_list = []
    # apply the evaluation function to all genomes
    for genome in genome_list:
        fitness = evaluate(genome)
        fitness_list.append(fitness)
        genome.SetFitness(fitness)

    # Here is where the author recommends to output information on the current generation
    top_fit = sorted(fitness_list)[-1]
    print("Highest fitness in generation " + str(generation) + ": " + str(top_fit))
    print("Percentage of Max Fitness: " + str(top_fit/2.0))

    # advance to the next generation
    pop.Epoch()
```

    Highest fitness in generation 0: 1.3847487012811497
    Percentage of Max Fitness: 0.6923743506405748
    Highest fitness in generation 1: 1.6269513980474455
    Percentage of Max Fitness: 0.8134756990237227
    Highest fitness in generation 2: 1.7014976657840726
    Percentage of Max Fitness: 0.8507488328920363
    Highest fitness in generation 3: 1.7805192458906676
    Percentage of Max Fitness: 0.8902596229453338
    Highest fitness in generation 4: 1.8412507847903352
    Percentage of Max Fitness: 0.9206253923951676
    Highest fitness in generation 5: 1.8990023622827406
    Percentage of Max Fitness: 0.9495011811413703
    Highest fitness in generation 6: 1.9354934634112784
    Percentage of Max Fitness: 0.9677467317056392
    Highest fitness in generation 7: 1.9416213769999735
    Percentage of Max Fitness: 0.9708106884999868
    Highest fitness in generation 8: 1.959301245633888
    Percentage of Max Fitness: 0.979650622816944
    Highest fitness in generation 9: 1.9799004840213486
    Percentage of Max Fitness: 0.9899502420106743

