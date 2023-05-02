def seed_everything(seed=1062):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# Functions and classes for the BRKGA-CNN
def createSolutionConv(in_feat, ConvLayer, genes):
    """
    The function decodes the individual's gene in a solution containing "ConvLayer"
    convolutional layers, Batch Normalization, Max-Pooling, and Activation function.
    """
    solConv = []

    for i in range(0, ConvLayer):

        # Convolutional Layer
        in_channel = in_feat if i == 0 else out_channel
        out_channel = int(np.round(genes[6 * i] * 20))
        if out_channel == 0:
            out_channel = 1
        kernelSize = int(np.round(genes[6 * i + 1] * 7))
        if kernelSize == 0:
            kernelSize = 1
        strid = 1 if genes[6 * i + 2] <= 0.5 else 2

        solConv.append(nn.Conv2d(in_channels=in_channel,
                                 out_channels=out_channel,
                                 kernel_size=(kernelSize, kernelSize),
                                 stride=(strid, strid), padding=(1, 1)))

        # Batch Normalization Layer
        if genes[6 * i + 3] <= 0.5:
            solConv.append(nn.BatchNorm2d(out_channel))

        # Max-Pooling Layer
        if genes[6 * i + 4] <= 0.5:
            solConv.append(nn.MaxPool2d(kernel_size=kernelSize,
                                        stride=(strid, strid)))

        # Activation Function for Convolutional Layer
        solConv.append(NonLinearActivation(np.round(16 * genes[6 * i + 5])))

    return solConv


def createSolutionLinear(in_feat, D_out, LinearLayer, genes):
    """
    The function decodes the individual's gene into a solution containing
    "LinearLayer" full-connected layers.
    """
    solLinear = []

    for i in range(0, LinearLayer):
        in_feat = in_feat if i == 0 else out_feat
        out_feat = int(np.round(genes[3 * i] * (in_feat - D_out) + D_out).item())

        solLinear.append(nn.Linear(in_features=in_feat,  # Linear layer input
                                   out_features=out_feat,  # Linear layer output
                                   bias=np.round(genes[3 * i + 1])))  # Bias: on (True) or off (False)
        # Activation function for Linear Layer
        solLinear.append(NonLinearActivation(np.round(16 * genes[3 * i + 2])))

    # The last linear layer contains the number of classes of the problem as output
    solLinear.append(nn.Linear(in_features=out_feat,
                               out_features=D_out,
                               bias=True))

    return solLinear  # np.array(solLinear, dtype=int)


class Model_cnn(torch.nn.Module):
  def __init__(self, model):
    super(Model_cnn, self).__init__()  
    self.layers_cnn = nn.Sequential(*model)

  def forward(self, x):
    x = self.layers_cnn(x)
    x = x.reshape(x.size(0), -1)
    return x.shape


class Model(torch.nn.Module):
  def __init__(self, cnn, fc):
    super(Model, self).__init__()
    self.layers_cnn = nn.Sequential(*cnn)
    self.layers_linear = nn.Sequential(*fc)
  
  def forward(self, x):
    x = self.layers_cnn(x)
    x = x.reshape(x.size(0), -1)
    x = self.layers_linear(x)
    return nn.functional.log_softmax(x, dim=1)


def train_cnn(solConv, val_loader):
    '''
    The function is responsible for training only the convolutional layers
    to determine how many output channels of the last layer will be used
    to represent the input of neurons in the linear layer.
    '''
    modelo_cnn = Model_cnn(solConv)
    modelo_cnn.to(device)
    modelo_cnn.train()
    for i, (images, labels) in enumerate(val_loader):
        if i == 0:
            img = images.to(device)
            outputs = modelo_cnn.forward(img.float())
        else:
            break
    return outputs[1]


def NonLinearActivation(number):
    if number == 0:
        return torch.nn.Sigmoid()
    elif number == 1:
        return torch.nn.Hardshrink(lambd=0.5)
    elif number == 2:
        return torch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None)
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
        return torch.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)
    elif number == 9:
        return torch.nn.ReLU6(inplace=False)
    elif number == 10:
        return torch.nn.SELU(inplace=False)
    elif number == 11:
        return torch.nn.CELU(alpha=1.0, inplace=False)
    elif number == 12:
        return torch.nn.ELU(alpha=1.0, inplace=False)
    elif number == 13:
        return torch.nn.Softplus(beta=1, threshold=20)
    elif number == 14:
        return torch.nn.Softshrink(lambd=0.5)
    elif number == 15:
        return torch.nn.Softsign()
    elif number == 16:
        return torch.nn.Tanh()

def train_model(solution, trained_models, initial_seed):
    if (str(solution)) in trained_models:
        return trained_models[str(solution)], trained_models

    else:
        seed_everything(initial_seed)
        sequential = Model(solution[0], solution[1])

        sequential.to(device)
        # print("Decoder:", sequential)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sequential.parameters(), lr=1e-2)

        # Creating the tensors
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=True, num_workers=0)

        results_valid = []
        for epoch in range(5):
            for inputs, targets in train_loader:
                x_train, y_train = inputs.to(device), targets.to(device)

                optimizer.zero_grad()  # Resetting the gradients due to the previous cycle
                y_pred = sequential.forward(
                    x_train.float())  # Forward pass: compute predicted y by passing x to the model
                y_train = y_train.clone().detach()  # .requires_grad_(True)   # y_train = torch.tensor(y_train, dtype=torch.long)
                loss = loss_fn(y_pred, y_train)  # Compute loss
                loss.backward(
                    retain_graph=True)  # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # Calling the step function on an Optimizer makes an update to its parameters

            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    Y_pred = sequential.forward(inputs.to(device).float())
                    labels = targets.clone().detach()  # .requires_grad_(True) #labels = torch.tensor(targets.to(device), dtype=torch.long)
                    _, predicted = torch.max(Y_pred.data, 1)
                    predicted = predicted.cpu()
                    total += labels.size(0)
                    correct += (predicted == targets).sum().item()
                results_valid.append(100.0 * (correct / total))

        # print(f"Accuracy: {results_valid} \n")
        error_valid = 100 - max(results_valid)
        trained_models[str(solution)] = error_valid
        return error_valid, trained_models


def test_model(best_model, initial_seed):  # train_loader, validation_loader, test_loader,
    seed_everything(initial_seed)
    sequential = Model(best_model[0], best_model[1])

    sequential.to(device)  # CUDA

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sequential.parameters(), lr=1e-2)

    # Creating the tensors (Tentar criar os tensores por fora e só rodar eles aqui)
    train_loader = data_utils.DataLoader(train_set, batch_size=100,
                                         shuffle=True)  # Cria um buffer para pegar os dados por partes
    validation_loader = data_utils.DataLoader(valid_set, batch_size=100,
                                              shuffle=True)  # Cria um buffer para pegar os dados por partes
    test_loader = data_utils.DataLoader(test_set, batch_size=100,
                                        shuffle=True)  # Cria um buffer para pegar os dados por partes

    train_acc, valid_acc, test_acc = [], [], []
    for epoch in range(100):
        for inputs, targets in train_loader:
            x_train, y_train = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Resetting the gradients due to the previous cycle
            y_pred = sequential.forward(x_train.float())  # Forward pass: compute predicted y by passing x to the model
            y_train = y_train.clone().detach()  # y_train = torch.tensor(y_train, dtype=torch.long)
            loss = loss_fn(y_pred, y_train)  # Compute loss
            loss.backward(
                retain_graph=True)  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Calling the step function on an Optimizer makes an update to its parameters

        # Testing Train Dataset
        correct1, total1 = 0, 0
        with torch.no_grad():  # Put all torch.no_grad requires in false
            for inputs, targets in train_loader:
                images, labels = inputs.to(device), targets.to(device)
                Y_pred = sequential.forward(images.float())  # Y_pred = sequential(inputs.to(device))
                labels = labels.clone().detach()  # labels = torch.tensor(labels, dtype=torch.long)
                _, predicted = torch.max(Y_pred.data, 1)
                predicted = predicted.cpu()
                total1 += labels.size(0)
                correct1 += (predicted == targets).sum().item()
            train_acc.append(100.0 * (correct1 / total1))

        # Testing Validation Dataset
        correct2, total2 = 0, 0
        with torch.no_grad():  # Put all torch.no_grad requires in false
            for inputs, targets in validation_loader:
                images, labels = inputs.to(device), targets.to(device)
                Y_pred = sequential.forward(images.float())  # Y_pred = sequential(inputs.to(device))
                labels = labels.clone().detach()  # labels = torch.tensor(labels, dtype=torch.long)
                _, predicted = torch.max(Y_pred.data, 1)
                predicted = predicted.cpu()
                total2 += labels.size(0)
                correct2 += (predicted == targets).sum().item()
            valid_acc.append(100.0 * (correct2 / total2))

        # Testing Test Dataset
        correct3, total3 = 0, 0
        with torch.no_grad():  # Put all torch.no_grad requires in false
            for inputs, targets in test_loader:
                images, labels = inputs.to(device), targets.to(device)
                Y_pred = sequential.forward(images.float())  # Y_pred = sequential(inputs.to(device))
                labels = labels.clone().detach()  # labels = torch.tensor(labels, dtype=torch.long)
                _, predicted = torch.max(Y_pred.data, 1)
                predicted = predicted.cpu()
                total3 += labels.size(0)
                correct3 += (predicted == targets).sum().item()
            test_acc.append(100.0 * (correct3 / total3))

    train_error = 100-max(train_acc)
    valid_error = 100-max(valid_acc)
    test_error = 100-max(test_acc)

    return train_error, valid_error, test_error
