def seed_everything(seed=1062):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# Functions and classes for the BRKGA-CNN
def createSolutionConv(in_feat, ConvLayer, genes, device):
    solConv = torch.zeros(8*ConvLayer, dtype=int, device=device) # Insert convolutional Layers

    for i in range(0, ConvLayer):

      D_out = torch.round(genes[6*i]*20)
      if D_out == 0:
        D_out = 1
      kernel_size = torch.round(genes[6*i+1]*D_out) if D_out <= 7 else torch.round(genes[6*i+1]*7)
      if kernel_size == 0:
        kernel_size = 1
      stride = 1 if genes[6*i+2] <= 0.5 else 2
      
      # Convolutional Layer
      solConv[8*i+0] = in_feat if i==0 else solConv[8*(i-1)+1]
      solConv[8*i+1] = D_out
      solConv[8*i+2] = kernel_size
      solConv[8*i+3] = stride
      # Batch Normalization
      if genes[6*i+3] <= 0.5:
        solConv[8*i+4] = D_out
      else:
        solConv[8*i+4] = 0
      # Max-Pooling
      if genes[6*i+4] <= 0.5:
        solConv[8*i+5] = kernel_size
        solConv[8*i+6] = stride
      else:
        solConv[8*i+5] = 0
        solConv[8*i+6] = 0
      # Activation Function for Convolutional Layer
      solConv[8*i+7] = torch.round(16 * genes[6*i+5])

    return solConv # np.array(solConv, dtype=int)

def createSolutionLinear(in_feat, D_out, LinearLayer, genes, device):
    solLinear = torch.zeros(4*LinearLayer, dtype=int, device=device)     # Insert Linear Layers    
    
    for i in range(0, LinearLayer):
      # Linear layer input
      solLinear[4*i] = in_feat if i==0 else solLinear[4*(i-1)+1]
      # Linear layer output
      solLinear[4*i+1] = torch.round(genes[3*i] * (solLinear[4*i] - D_out) + D_out)
      # Bias: on (True) or off (False)
      solLinear[4*i+2] = torch.round(genes[3*i+1])
      # Activation function for Linear Layer
      solLinear[4*i+3] = torch.round(16 * genes[3*i+2])

    return solLinear # np.array(solLinear, dtype=int)


def createNetConv(solConv, MaxConv):
  auxModel = []
  for i in range(0, MaxConv):
    chromo = solConv[8*i:(8*i)+8]
    auxModel.append(nn.Conv2d(in_channels=chromo[0],
                              out_channels=chromo[1],
                              kernel_size=(chromo[2], chromo[2]),
                              stride=(chromo[3], chromo[3]), padding=1))
    if chromo[4] != 0:
      auxModel.append(nn.BatchNorm2d(chromo[4]))
    if chromo[5] != 0:
      auxModel.append(nn.MaxPool2d(kernel_size=chromo[5],
                                   stride=(chromo[6], chromo[6]))) # chromo[5:7] torch.max_pool2d
    auxModel.append(NonLinearActivation(chromo[7]))
  return auxModel

def createNetLinear(solLinear, MaxLinear):
  auxModel = []
  for i in range(0, MaxLinear):
      auxModel.append(nn.Linear(in_features=solLinear[4*i],
                                out_features=solLinear[4*i+1],
                                bias=solLinear[4*i+2]
                                )
                      )
      auxModel.append(NonLinearActivation(solLinear[4*i+3]))    
  auxModel.append(nn.Linear(in_features=solLinear[-3], out_features=D_out, bias=True))
  return auxModel

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
    return nn.functional.log_softmax(x, dim=1) # nn.functional.log_softmax(self.layers_linear(x), dim=1) # self.fout(x)  # torch.sigmoid(self.fout(x))

def train_cnn(solConv, val_loader, ConvLayer):
  '''
  The function is responsible for training only the convolutional layers
  to determine how many output channels of the last layer will be used
  to represent the input of neurons in the linear layer. 
  '''
  auxModel = createNetConv(solConv.tolist(), ConvLayer)
  # seed_everything(seed=2022)
  
  modelo_cnn = Model_cnn(auxModel)
  # print("Model CNN:", modelo_cnn)
  modelo_cnn.to(device)
  modelo_cnn.train()
  for i, (images, labels) in enumerate(val_loader):
    if i == 0:
      img = images.to(device)
      outputs = modelo_cnn.forward(img.float())
    else:
      break
  #print("Input of Layer Linear:", outputs[1])
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

def train_model(solution, ConvLayer, LinearLayer, trained_models, initial_seed):

    auxModel_cnn = createNetConv(solution[0], ConvLayer)
    auxModel_linear = createNetLinear(solution[1], LinearLayer)
  
    if (str(solution)) in trained_models:
        return [auxModel_cnn, auxModel_linear], trained_models[str(solution)], trained_models
    else:

      seed_everything(initial_seed)
      sequential = Model(auxModel_cnn, auxModel_linear)
      
      sequential.to(device)
      # print("Decoder:", sequential)

      loss_fn = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(sequential.parameters(), lr=1e-2)
      
      # Creating the tensors
      train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True) #, num_workers=0)
      val_loader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=True) #, num_workers=0)
      
      results_valid = []
      for epoch in range(20):
        for inputs, targets in train_loader:
          x_train, y_train = inputs.to(device), targets.to(device)
          
          optimizer.zero_grad()                               # Resetting the gradients due to the previous cycle
          y_pred = sequential.forward(x_train.float())        # Forward pass: compute predicted y by passing x to the model
          # y_train = torch.tensor(y_train, dtype=torch.long)   # sourceTensor.clone().detach()
          loss = loss_fn(y_pred, y_train)                     # Compute loss
          loss.backward(retain_graph=True)                    # Backward pass: compute gradient of the loss with respect to model parameters
          optimizer.step()                                    # Calling the step function on an Optimizer makes an update to its parameters

        correct, total = 0, 0
        with torch.no_grad():
          for inputs, targets in val_loader:
            images, labels = inputs.to(device), targets.to(device)
            Y_pred = sequential.forward(images.float())
            # labels = torch.tensor(labels, dtype=torch.long)   # sourceTensor.clone().detach()
            _, predicted = torch.max(Y_pred.data, 1)
            predicted = predicted.cpu()
            total += labels.size(0)
            correct += (predicted == targets).sum().item()
          results_valid.append(100.0 * (correct / total))
      
      error_valid = 100 - np.max(results_valid)
      
      # print(f"Error: {100-results_valid} \n")
      trained_models[str(solution)] = error_valid # 100-results_valid
      return [auxModel_cnn, auxModel_linear], error_valid, trained_models


def test_model(best_model, seed): # train_loader, validation_loader, test_loader, 
  seed_everything(seed)

  sequential = Model(best_model[0], best_model[1])
  sequential.to(device) # CUDA
  
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(sequential.parameters(), lr=1e-2)
  
  # Creating the tensors
  train_loader = data_utils.DataLoader(train_set, batch_size=100, shuffle=True) #, num_workers=0) # Cria um buffer para pegar os dados por partes
  validation_loader = data_utils.DataLoader(valid_set, batch_size=100, shuffle=True) #, num_workers=0) # Cria um buffer para pegar os dados por partes
  test_loader = data_utils.DataLoader(test_set, batch_size=100, shuffle=True) #, num_workers=0) # Cria um buffer para pegar os dados por partes

  train_acc, valid_acc, test_acc = [], [], []

  for epoch in range(200):
    for inputs, targets in train_loader:
      x_train, y_train = inputs.to(device), targets.to(device)
      
      optimizer.zero_grad()               # Resetting the gradients due to the previous cycle
      y_pred = sequential.forward(x_train.float())        # Forward pass: compute predicted y by passing x to the model
      loss = loss_fn(y_pred, y_train)     # Compute loss
      loss.backward(retain_graph=True)    # Backward pass: compute gradient of the loss with respect to model parameters
      optimizer.step()                    # Calling the step function on an Optimizer makes an update to its parameters

    # Testing Train Dataset
    correct, total = 0, 0
    with torch.no_grad(): # Put all torch.no_grad requires in false
      for inputs, targets in train_loader:
        Y_pred = sequential.forward(inputs.to(device).float())
        _, predicted = torch.max(Y_pred.data, 1)
        predicted = predicted.cpu()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
      train_acc.append(100.0 * (correct / total))

    # Testing Validation Dataset
    correct, total = 0, 0
    with torch.no_grad(): # Put all torch.no_grad requires in false
      for inputs, targets in validation_loader:
        Y_pred = sequential.forward(inputs.to(device).float())
        _, predicted = torch.max(Y_pred.data, 1)
        predicted = predicted.cpu()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
      valid_acc.append(100.0 * (correct / total))

    # Testing Test Dataset
    correct, total = 0, 0
    with torch.no_grad(): # Put all torch.no_grad requires in false
      for inputs, targets in test_loader:
        Y_pred = sequential.forward(inputs.to(device).float())
        _, predicted = torch.max(Y_pred.data, 1)
        predicted = predicted.cpu()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
      test_acc.append(100.0 * (correct / total))
  
  train_error = 100 - np.max(train_acc)
  valid_error = 100 - np.max(valid_acc)
  test_error = 100 - np.max(test_acc)
  print(f"Error: \nTrain: {np.round(train_error, 4)} - Valid: {np.round(valid_error,4)} - Test:{np.round(test_error, 4)}")
  return train_error, valid_error, test_error

def individualCorrection(solConv, val_loader, layer):
    error1 = 'Given input size:'
    error2 = 'Calculated padded input size per channel:'
    last_layer = False
    for index in range(layer-1, -1, -1):
        try:
            in_f = train_cnn(solConv, val_loader, layer)
            return in_f
            break
        except RuntimeError as err:
          err = str(err)
          # print("\nError:", err)
          
          if err[:17] == error1:
              for index1 in range(layer-1, -1, -1):
                  if solConv[8*index1+5] != 0:
                      # print("Erro do Max-Pooling:")
                      solConv[8*index1+5] = 0
                      solConv[8*index1+6] = 0
                      # print("Changed SolConv:", solConv)

                      try:
                          in_f = train_cnn(solConv, val_loader, layer)
                          return in_f
                          break
                      except:
                          pass
              
          elif err[:41] == error2 and last_layer==False:
              # print("Erro do kernel size greather output channel:")
              #print(f"Output Channel: {err[43]} - Kernel size: {err[65]}")
              #out_channel = int(err[43])
              #kernel_size = int(err[65])
              for i in range(0, layer):
                  if  solConv[8*i+1] <= solConv[8*i+2]:
                      # print(solConv[8*i+1], solConv[8*i+2])
                      if solConv[8*i+1] > 1:
                          solConv[8*i+2] = torch.randint(1, solConv[8*i+1], (1,))
                      else:
                          solConv[8*i+2] = 1
                      # print("Solução modificada pelo kernel size:", solConv)
                  elif i == layer-1:
                      solConv[8*i+2] = torch.tensor([1])
              last_layer=True

          elif err[:41] == error2 and last_layer==True:
              # print("Erro de kernel size com problema de Max-Pooling:")
              for index2 in range(layer-1, -1, -1):

                  if solConv[8*index2+2] == int(err[65]):
                      # print("Corrigindo o kernel com o valor da input channel:")
                      solConv[8*index2+2] = int(err[43])

                  if solConv[8*index2+5] != 0:
                      solConv[8*index2+5] = 0
                      solConv[8*index2+6] = 0
                      try:
                          in_f = train_cnn(solConv, val_loader, layer)
                          return in_f
                          break
                      except:
                          pass
                      # print("Changed SolConv:", solConv)
          else:
            print("\nErro não solucionável!")
            return False

    in_f = train_cnn(solConv, val_loader, layer)
    return in_f

