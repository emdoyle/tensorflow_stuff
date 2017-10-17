import MultiNEAT as NEAT

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

# The Population constructor (again in C++) is copied below
# Population::Population(const Genome& a_Seed, const Parameters& a_Parameters,
#     bool a_RandomizeWeights, double a_RandomizationRange, int a_RNG_seed)
pop = NEAT.Population(genome, params, True, 1.0, 0)

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

for generation in range(10): # run for 100 generations

    # retrieve a list of all genomes in the population
    genome_list = NEAT.GetGenomeList(pop)

    fitness_list = []
    # apply the evaluation function to all genomes
    for genome in genome_list:
        fitness = evaluate(genome)
        fitness_list.append(fitness)
        genome.SetFitness(fitness)

    # Here is where the author recommends to output information on the current generation
    print("Highest fitness in generation " + str(generation) + ": " + str(sorted(fitness_list)[-1]))

    # advance to the next generation
    pop.Epoch()