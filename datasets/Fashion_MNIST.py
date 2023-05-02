import torch # for Deep Learning
from torchvision import datasets
import numpy as np
import torch.utils.data as data_utils


# Separate Data into Training, Validation and Testing
def dataset(return_tensor=True):
    # Use standard FashionMNIST dataset
    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    x_train = [item.reshape(1,28*28) for item in train_set.data]
    X_train = [np.array(x_train[n][0]) for n in range(len(x_train))]
    x_test = [item.reshape(1,28*28) for item in test_set.data]
    X_test = [np.array(x_test[n][0]) for n in range(len(x_test))]

    x_train = np.array(X_train)
    xtest = np.array(X_test)
    y_train = np.array(train_set.targets)
    ytest = np.array(test_set.targets)

    x_valid, x_test, y_valid, y_test = train_test_split(xtest, ytest, test_size=.50, random_state=1)

    input_shape = (1, 28, 28)
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_valid = x_valid.reshape(x_valid.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)

    x_train = torch.stack([torch.from_numpy(np.array(i)) for i in x_train])
    x_valid = torch.stack([torch.from_numpy(np.array(i)) for i in x_valid])
    x_test = torch.stack([torch.from_numpy(np.array(i)) for i in x_test])

    y_train = torch.stack([torch.from_numpy(np.array(i)) for i in y_train])
    y_valid = torch.stack([torch.from_numpy(np.array(i)) for i in y_valid])
    y_test = torch.stack([torch.from_numpy(np.array(i)) for i in y_test])

    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    valid_set = torch.utils.data.TensorDataset(x_valid, y_valid)
    test_set = torch.utils.data.TensorDataset(x_test, y_test)

    val_loader = torch.utils.data.DataLoader(valid_set,   batch_size=100, shuffle=True)
    
    return train_set, valid_set, test_set, val_loader

  
