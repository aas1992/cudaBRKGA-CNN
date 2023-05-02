from dataclasses import dataclass

import numpy as np

# from pyasn1.type.univ import Any
from typing import Any

# Carregando todas as bibliotecas necessárias
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

from keras.datasets import mnist  # MNIST dataset is included in Keras

from time import time
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils

from random import random, randint
import math
import copy
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using PyTorch version:", torch.__version__, " Device:", device)

# Use standard MNIST dataset
(xtrain, y_train), (xtest, ytest) = mnist.load_data()

x_train = [item.reshape(1, 28 * 28) for item in xtrain]
X_train = [x_train[n][0] for n in range(len(x_train))]
x_test = [item.reshape(1, 28 * 28) for item in xtest]
X_test = [x_test[n][0] for n in range(len(x_test))]

x_train = np.array(X_train)
X_test = np.array(X_test)

x_valid, x_test, y_valid, y_test = train_test_split(
    X_test, ytest, test_size=0.10, random_state=1
)


# # CONVERTING THE DATA TO TENSORS
input_shape = (1, 28, 28)
batch_size = 100

x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_valid = x_valid.reshape(x_valid.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

x_train = torch.stack([torch.from_numpy(np.array(i)) for i in x_train])
print(x_train.shape)
x_valid = torch.stack([torch.from_numpy(np.array(i)) for i in x_valid])
print(x_valid.shape)
x_test = torch.stack([torch.from_numpy(np.array(i)) for i in x_test])
print(x_test.shape)

y_train = torch.stack([torch.from_numpy(np.array(i)) for i in y_train])
print(y_train.shape)
y_valid = torch.stack([torch.from_numpy(np.array(i)) for i in y_valid])
print(y_valid.shape)
y_test = torch.stack([torch.from_numpy(np.array(i)) for i in y_test])
print(y_test.shape)

train_set = torch.utils.data.TensorDataset(x_train, y_train)
valid_set = torch.utils.data.TensorDataset(x_valid, y_valid)
test_set = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=0
)


import brkga


def seed_everything(seed=1062):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decoder(ind, MaxConv, MaxLinear):
    seed_everything(2022)
    model_cnn, model_linear = [], []
    auxModel_cnn = createNetConv(model_cnn, ind.genes[0], MaxConv)
    auxModel_linear = createNetLinear(model_linear, ind.genes[1], MaxLinear)
    sequential = Model(auxModel_cnn, auxModel_linear)
    ind.model = sequential
    sequential.to(device)
    print("Decoder:", sequential)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sequential.parameters(), lr=1e-2)

    # Creating the tensors
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    loss_train = []
    for t in range(30):
        for inputs, targets in train_loader:
            x_train, y_train = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Resetting the gradients due to the previous cycle
            y_pred = sequential.forward(
                x_train.float()
            )  # Forward pass: compute predicted y by passing x to the model
            y_train = torch.tensor(y_train, dtype=torch.long)
            loss = loss_fn(y_pred, y_train)  # Compute loss
            loss_train.append(loss.cpu().detach().item())
            loss.backward(
                retain_graph=True
            )  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Calling the step function on an Optimizer makes an update to its parameters

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            images, labels = inputs.to(device), targets.to(device)
            Y_pred = sequential.forward(images.float())
            labels = torch.tensor(labels, dtype=torch.long)
            _, predicted = torch.max(Y_pred.data, 1)
            predicted = predicted.cpu()
            total += labels.size(0)
            correct += (predicted == targets).sum().item()
        results_valid = 100.0 * (correct / total)

    # valid_error = 100 - results_valid      #train_average = np.mean(loss_train)
    print("Error:", 100 - results_valid, "Loss:", np.mean(loss_train))
    print()
    ind.fitness, ind.loss = 100 - results_valid, np.mean(loss_train)
    return ind


# Functions and Classes for BRKGA-CNN
def createIndividualCNN(
    in_feat, N
):  # , conv_previous - Na verdade, 'al' é pra ser igual ao número de camadas ocultas para camadas lineares
    chromossome = [random() for i in range(0, 4 * N)]  # Insert Convolutional Layers

    stride = 1 if chromossome[0] < 0.5 else 2
    D_out_1 = math.floor(chromossome[0] * 10)
    D_out = D_out_1 if D_out_1 > 0 else 1
    chromossome[0] = [
        in_feat,
        D_out,
        randint(1, 5),
        stride,
        (1, 1),
    ]  # Convolutional Layer
    chromossome[1] = D_out if chromossome[1] >= 0.5 else None  # Batch Normalization
    chromossome[2] = (
        [randint(1, 5), (stride, stride)] if chromossome[2] >= 0.5 else None
    )  # Max-Pooling random.randint(1, 2), (1, 1)
    chromossome[3] = randint(0, 16)  # Activation Function for Convolutional Layer

    if N > 1:
        for i in range(1, N):
            stride = 1 if chromossome[4 * i] < 0.5 else 2
            D_out_1 = math.floor(chromossome[4 * i] * 10)
            D_out = D_out_1 if D_out_1 > 0 else 1
            chromossome[4 * i] = [
                chromossome[4 * (i - 1)][1],
                D_out,
                randint(1, 5),
                stride,
                (1, 1),
            ]  # Convolutional Layer
            chromossome[4 * i + 1] = (
                D_out if chromossome[4 * i + 1] >= 0.5 else None
            )  # Batch Normalization
            chromossome[4 * i + 2] = (
                [randint(1, 5), (stride, stride)]
                if chromossome[4 * i + 2] >= 0.5
                else None
            )  # Max-Pooling random.randint(1, 2), (1, 1)
            chromossome[4 * i + 3] = randint(
                0, 16
            )  # Activation Function for Convolutional Layer
    return chromossome


def createIndividualLinear(D_out, N, in_feat=None):  # model
    chromo_linear = 4 * N * [None]  # Insert Linear Layers
    chromo_linear[0] = in_feat  # Linear layer input
    if in_feat >= D_out:
        chromo_linear[1] = randint(D_out, in_feat)
    else:
        chromo_linear[1] = D_out  # Linear layer output
    chromo_linear[2] = (
        True if random() >= 0.5 else False
    )  # Bias on (True) or off (False)
    chromo_linear[3] = randint(0, 16)  # Activation function

    if N > 1:  # Se existir mais de uma camada linear
        for i in range(1, N):
            chromo_linear[4 * i] = chromo_linear[4 * (i - 1) + 1]  # Linear layer input
            chromo_linear[4 * i + 1] = (
                randint(D_out, chromo_linear[4 * i])
                if chromo_linear[4 * i] < D_out
                else D_out
            )  # Linear layer output
            chromo_linear[4 * i + 2] = (
                True if random() >= 0.5 else False
            )  # Bias on (True) or off (False)
            chromo_linear[4 * i + 3] = randint(0, 16)  # Activation function
    return chromo_linear


def NonLinearActivation(number):
    if number == 0:
        return torch.nn.ELU(alpha=1.0, inplace=False)
    elif number == 1:
        return torch.nn.Hardshrink(lambd=0.5)
    elif number == 2:
        return torch.nn.Hardtanh(
            min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None
        )
    elif number == 3:
        return torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
    elif number == 4:
        return torch.nn.LogSigmoid()
    elif number == 5:
        return torch.nn.Tanhshrink()
    elif number == 6:
        return torch.nn.PReLU(num_parameters=1, init=0.25)
    elif number == 7:
        return torch.nn.ReLU(inplace=False)
    elif number == 8:
        return torch.nn.RReLU(lower=0.125, upper=0.34, inplace=False)
    elif number == 9:
        return torch.nn.ReLU6(inplace=False)
    elif number == 10:
        return torch.nn.SELU(inplace=False)
    elif number == 11:
        return torch.nn.CELU(alpha=1.0, inplace=False)
    elif number == 12:
        return torch.nn.Sigmoid()
    elif number == 13:
        return torch.nn.Softplus(beta=1, threshold=20)
    elif number == 14:
        return torch.nn.Softshrink(lambd=0.5)
    elif number == 15:
        return torch.nn.Softsign()
    elif number == 16:
        return torch.nn.Tanh()
    elif number == 17:
        return torch.nn.functional.log_softmax()


def createNetConv(model, chromossome, MaxConv):
    auxModel = list.copy(model)
    for i in range(0, MaxConv):
        auxModel.append(nn.Conv2d(*chromossome[4 * i]))
        if chromossome[4 * i + 1]:
            auxModel.append(nn.BatchNorm2d(chromossome[4 * i + 1]))
        if chromossome[4 * i + 2]:
            auxModel.append(nn.MaxPool2d(*chromossome[4 * i + 2]))  # torch.max_pool2d
        auxModel.append(NonLinearActivation(chromossome[4 * i + 3]))
    return auxModel


def createNetLinear(model, chromo_linear, N):  # ACHO QUE O PROBLEMA ESTÁ AQUI
    auxModel = list.copy(model)
    for i in range(0, N - 1):
        auxModel.append(nn.Linear(*chromo_linear[4 * i : 4 * i + 3]))
        auxModel.append(NonLinearActivation(chromo_linear[4 * i + 3]))
        auxModel.append(nn.Linear(chromo_linear[4 * i + 1], D_out, True))
    return auxModel


class Model_cnn(torch.nn.Module):
    def __init__(self, model):
        super(Model_cnn, self).__init__()
        self.layers_cnn = nn.Sequential(*model)

    def forward(self, x):
        x = self.layers_cnn(x)
        x = x.reshape(x.size(0), -1)
        return x.shape


def train_cnn(chromossome, val_loader, MaxConv):
    """
  The function is responsible for training only the convolutional layers
  to determine how many output channels of the last layer will be used
  to represent the input of neurons in the linear layer. 
  """
    model = []
    auxModel = createNetConv(model, chromossome, MaxConv)
    seed_everything(seed=2022)
    modelo_cnn = Model_cnn(auxModel)
    modelo_cnn.to(device)
    modelo_cnn.train()
    for i, (images, labels) in enumerate(val_loader):
        if i == 0:
            img = images.to(device)
            outputs = modelo_cnn.forward(img.float())
        else:
            break
    # print("Input of Layer Linear:", outputs[1])
    return outputs[1]


class Model(torch.nn.Module):
    def __init__(self, cnn, fc):
        super(Model, self).__init__()
        self.layers_cnn = nn.Sequential(*cnn)
        self.layers_linear = nn.Sequential(*fc)

    def forward(self, x):
        x = self.layers_cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.layers_linear(x)
        return nn.functional.log_softmax(
            x, dim=1
        )  # nn.functional.log_softmax(self.layers_linear(x), dim=1) # self.fout(x)  # torch.sigmoid(self.fout(x))


class Individual:
    def __init__(self, list_cnn, list_linear):
        self.genes = [list_cnn, list_linear]
        self.fitness = None
        self.loss = None


def createIndividual(in_feat_cnn, D_out, MaxLinear, MaxConv, Size):
    """
  The function is responsible for randomly creating i individuals (based on size)
  containing convolutional and linear layers separated by a simple list.
  """
    list_individuals = [None for i in range(Size)]
    for i in range(0, Size):
        model = []
        lista_conv = createIndividualCNN(
            in_feat_cnn, MaxConv
        )  # Criar um cromossomo com genes convolucionais aleatórios
        in_f = train_cnn(
            lista_conv, val_loader, MaxConv
        )  # Treinar a rede convolucional para saber a saída da última camada para ser a entrada da camada linear
        lista_linear = createIndividualLinear(
            D_out, MaxLinear, in_feat=in_f
        )  # Criar um cromossomo com genes lineares, a partir da última camada convol.
        # print(lista_conv, lista_linear)
        list_individuals[i] = Individual(lista_conv, lista_linear)
    return list_individuals


@dataclass
class CNNIndividual(brkga.Individual):
    genes: Any = None
    model: Any = None
    fitness: Any = None
    loss: Any = None


class BrkgaCNN(brkga.DefaultBRKGA):
    def __init__(self, *args, model, MaxConv, MaxLinear, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.MaxConv = MaxConv
        self.MaxLinear = MaxLinear

    def create_single_individual(self, **kwargs) -> brkga.Individual:
        lista_conv = createIndividualCNN(
            in_feat_cnn, self.MaxConv
        )  # Criar um cromossomo com genes convolucionais aleatórios
        in_f = train_cnn(
            lista_conv, val_loader, self.MaxConv
        )  # Treinar a rede convolucional para saber a saída da última camada para ser a entrada da camada linear
        lista_linear = createIndividualLinear(
            D_out, self.MaxLinear, in_feat=in_f
        )  # Criar um cromossomo com genes lineares, a partir da última camada convol.
        # print(lista_conv, lista_linear)
        genes = [lista_conv, lista_linear]
        print("genes:", genes)
        return CNNIndividual(genes=genes, model=self.model)

    # Decoder
    def calculate_fitness(self, ind, **fitness_args):
        ind = decoder(ind, self.MaxConv, self.MaxLinear)
        return ind.fitness


in_feat_linear = 20  # Linear Layer Input Features.
in_feat_cnn = 1  # Convolutional Layer Input Features.
D_out = 10  # Output Features.
MaxLinear = 2  # Maximum number of Linear Layers
MaxConv = 2  # Maximum number of Convolutional Layers


if __name__ == "__main__":
    inicio = time()  # Contador inicial do tempo de resolução
    SEED = 2021  # Semente fixada
    model = []  # Model com as camadas
    best_ind = None  # Melhor indivíduo

    for i, j in zip(range(1, MaxLinear + 1), range(1, MaxConv + 1)):
        cnn = BrkgaCNN(
            pop_size=10,
            mutant_size=0.3,
            elite_size=0.3,
            qtd_generations=5,
            initial_seed=SEED,
            rho_e=0.7,
            maximize=False,
            model=model,
            MaxConv=i,
            MaxLinear=j,
        )
        cnn.run()
        best_ind = (
            cnn.best_ind if best_ind is None else cnn.best(best_ind, cnn.best_ind)
        )

        print("Melhor Indivíduo Local para {} camada(s) oculta(s):".format(i))
        print("Genes:", cnn.best_ind.genes)
        print("Fitness:", cnn.best_ind.fitness)
        print("Loss:", cnn.best_ind.loss)
        print("Model:", cnn.best_ind.model)

        # model = cnn.best_ind.model

    print("\nMelhor Indivíduo:")
    print("Genes:", best_ind.genes)
    print("Fitness:", best_ind.fitness)
    print("Loss:", best_ind.loss)
    print("Model:", best_ind.model)

    print("\nTraining time (in min) =", (time() - inicio) / 60)
