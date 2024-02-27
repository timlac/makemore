import torch
import torch.nn.functional as F
from batch_norm_makemore.utils import build_dataset, create_stoi
from batch_norm_makemore.initialization import initialize_parameters
from batch_norm_makemore.models import Tanh
import random
import matplotlib.pyplot as plt


g = torch.Generator().manual_seed(2147483647)

random.seed(99)

words = open('../names.txt', 'r').read().splitlines()

stoi, itos = create_stoi(words)
vocab_size = len(itos)

# build the dataset
block_size = 3  # context length: how many characters do we take to predict the next one?

random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)  # 80%
Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)  # 10%
Xte, Yte = build_dataset(words[n2:], block_size, stoi)  # 10%

C, layers, parameters = initialize_parameters(vocab_size, block_size)

# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]  # batch X,Y

    # forward pass
    emb = C[Xb]  # embed the characters into vectors
    x = emb.view(emb.shape[0], -1)  # concatenate the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)  # loss function

    # backward pass
    for layer in layers:
        layer.out.retain_grad()  # AFTER_DEBUG: would take out retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 150000 else 0.01  # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:  # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr * p.grad).std() / p.data.std()).log10().item() for p in parameters])

    if i >= 1000:
        break  # AFTER_DEBUG: would take out obviously to run full optimization