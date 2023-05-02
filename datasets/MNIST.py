import torch # for Deep Learning
import numpy as np
import torch.utils.data as data_utils
from keras.datasets import mnist     # MNIST dataset is included in Keras

# Separate Data into Training, Validation and Testing
def dataset(return_tensor=True):
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    x_train = [item.reshape(1,28*28) for item in xtrain]
    X_train = [x_train[n][0] for n in range(len(x_train))]
    x_test = [item.reshape(1,28*28) for item in xtest]
    X_test = [x_test[n][0] for n in range(len(x_test))]

    x_train = np.array(X_train)
    X_test = np.array(X_test)

    x_valid, x_test, y_valid, y_test = train_test_split(X_test, ytest, test_size=.50, random_state=1)

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
  
