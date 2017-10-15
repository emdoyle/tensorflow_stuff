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

![caption](files/PreDetermTopo.png)

    2. Start with a very simple network consisting of only input and output layers and evolve new nodes
    and connections along with connection weights and biases.

![caption](files/EvolvedTopo.png)
