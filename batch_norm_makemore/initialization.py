import torch
from batch_norm_makemore.models import Linear, BatchNorm1d, Tanh


def initialize_parameters(vocab_size, block_size):
    n_embd = 10  # the dimensionality of the character embedding vectors
    n_hidden = 100  # the number of neurons in the hidden layer of the MLP
    g = torch.Generator().manual_seed(2147483647)  # for reproducibility

    C = torch.randn((vocab_size, n_embd), generator=g)

    layers = [
        Linear(n_embd * block_size, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), Tanh(),
        Linear(n_hidden, n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ]

    with torch.no_grad():
        # last layer: make less confident
        layers[-1].weight *= 0.1
        # layers[-1].weight *= 0.1
        # all other layers: apply gain
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 1.0  # 5/3

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(sum(p.nelement() for p in parameters))  # number of parameters in total
    for p in parameters:
        p.requires_grad = True

    return C, layers, parameters
