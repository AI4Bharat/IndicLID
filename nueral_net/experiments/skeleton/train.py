# Neural Net
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim


confusion_matrix_mapping  = {
    'Assamese' : 0,
    'Bangla' : 1,
    'Bodo' : 2,
    'Konkani' : 3, 
    'Gujarati' : 4,
    'Hindi' : 5,
    'Kannada' : 6,
    'Kashmiri' : 7,
    'Maithili' : 8,
    'Malayalam' : 9,
    'Manipuri_Mei' : 10,
    'Marathi' : 11,
    'Nepali' : 12,
    'Oriya' : 13,
    'Punjabi' : 14,
    'Sanskrit' : 15,
    'Sindhi' : 16,
    'Tamil' : 17,
    'Telugu' : 18,
    'Urdu' : 19
}

confusion_matrix_reverse_mapping  = {
    0 : 'Assamese',
    1 : 'Bangla',
    2 : 'Bodo',
    3 : 'Konkani', 
    4 : 'Gujarati',
    5 : 'Hindi',
    6 : 'Kannada',
    7 : 'Kashmiri',
    8 : 'Maithili',
    9 : 'Malayalam',
    10 : 'Manipuri_Mei',
    11 : 'Marathi',
    12 : 'Nepali',
    13 : 'Oriya',
    14 : 'Punjabi',
    15 : 'Sanskrit',
    16 : 'Sindhi',
    17 : 'Tamil',
    18 : 'Telugu',
    19 :  'Urdu'
}




train_tensor = torch.load('../corpus/tensor_train.pt')
train_X, train_y = train_tensor[:,:-1], train_tensor[:,-1]

test_tensor = torch.load('../corpus/tensor_test.pt')
test_X, test_y = test_tensor[:,:-1], test_tensor[:,-1]

valid_tensor = torch.load('../corpus/tensor_valid.pt')
valid_X, valid_y = valid_tensor[:,:-1], valid_tensor[:,-1]



class DATA(Dataset):
    def __init__(self, X, Y):
        self.size = len(Y)
        self.x = X
        self.y = Y

    def __len__(self):
        return (self.size)

    def __getitem__(self, idx):
        #print(self.x[idx].shape,self.y[idx].shape )
        return self.x[idx].float(), self.y[idx]





def get_dataloaders(X_train, X_test, X_valid, y_train, y_test, y_valid, batch_size):
    X_train, X_test, X_valid, y_train, y_test, y_valid = X_train, X_test, X_valid, torch.tensor(y_train, dtype = torch.long), torch.tensor(y_test, dtype = torch.long), torch.tensor(y_valid, dtype = torch.long)

    train = DATA(X_train,y_train)
    test = DATA(X_test,y_test)
    valid = DATA(X_valid,y_valid)


    train_dl = torch.utils.data.DataLoader(train,
                                                batch_size=batch_size,
                                                shuffle=True
                                            )

    test_dl = torch.utils.data.DataLoader(test,
                                                batch_size=len(y_test),
                                                shuffle=False
                                            )

    valid_dl = torch.utils.data.DataLoader(valid,
                                                batch_size=len(y_valid),
                                                shuffle=False
                                            )

    return train_dl, test_dl, valid_dl





train_dataloader, test_dataloader, valid_dataloader = get_dataloaders(train_X, test_X, valid_X, train_y, test_y, valid_y, batch_size = 32)
# n_batches = len(train_dataloader)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 24)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = Net()
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.003)







train_acc, valid_acc = [], []
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        
        # labels = list(labels.view(1, -1))  +
        # print(labels.shape)
        
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # labels = torch.squeeze(labels)

        # print(inputs.shape)
        # print(labels.shape)

        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
          print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
          running_loss = 0.0
    
    
    
    
    correct = 0
    total = 0
  
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in valid_dataloader:
            inputs, labels = data
            # calculate outputs by running inputs through the network
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            
            # debug()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    valid_acc.append(100 * correct / total)
    print(f'Accuracy of the network on the {total} valid inputs: {100 * correct / total} ')
    
    
    
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in train_dataloader:
            inputs, labels = data
            # calculate outputs by running inputs through the network
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            
            # debug()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    train_acc.append(100 * correct / total)
    print(f'Accuracy of the network on the {total} train inputs: {100 * correct / total} %')
    
print('Finished Training')
print(max(train_acc), max(test_acc))



torch.save(net, '../result/basline_nn_simple.pt')

model_scripted = torch.jit.script(net) # Export to TorchScript
model_scripted.save('../result/basline_nn_jit.pt')
