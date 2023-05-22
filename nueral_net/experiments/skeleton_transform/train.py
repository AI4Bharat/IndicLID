import fasttext
from tqdm import tqdm
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim


max_length = 30
input_dimension = 512

# embed_model = fasttext.train_unsupervised(
#         input = '../corpus/train_combine.txt',
#         minn=3, 
#         maxn=6, 
#         dim=input_dimension,
#         model='skipgram',
#         lr=0.1
#         )

# embed_model.save_model("../result/embed.bin")

embed_model = fasttext.load_model('../result/embed.bin')

def bin_to_vec(embed_model):
    
    # original BIN model loading
    f = embed_model

    # get all words from model
    words = f.get_words()

    with open('../result/embed.vec','w') as file_out:
        
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in words:
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr+'\n')
            except:
                pass


# embed_model_path = "../result/embed.bin"
# bin_to_vec(fasttext.load_model(embed_model_path))
# bin_to_vec(embed_model)


# data loading
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


df_train = pd.read_csv('../corpus/train_combine.txt', sep="\t",  header = None)
df_test = pd.read_csv('../corpus/test_combine.txt', sep="\t", quoting=csv.QUOTE_NONE,  header = None)
df_valid = pd.read_csv('../corpus/valid_combine.txt', sep="\t",  header = None)

print(df_train.shape)
train_X = df_train.iloc[:, 1]
train_y = df_train.iloc[:, 0]

test_X = df_test.iloc[:, 1]
test_y = df_test.iloc[:, 0]

valid_X = df_valid.iloc[:, 1]
valid_y = df_valid.iloc[:, 0]


class DATA(Dataset):
    def __init__(self, X, Y, embed_model_dim, embed_model, max_length, transform = None):
        self.size = len(Y)
        self.x = X
        self.y = Y
        self.transform = transform
        self.embed_model_dim = embed_model_dim
        self.embed_model = embed_model
        
        self.max_length = max_length

    def __len__(self):
        return (self.size)

    def __getitem__(self, idx):
        #print(self.x[idx].shape,self.y[idx].shape )
        sample = self.x[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample, self.embed_model_dim, self.embed_model, self.max_length)
        
        return sample

class WordEmbeding():
    def __call__(self, sample, embed_model_dim, embed_model, max_length):
        sen, label = sample
        
        sen = ' '.join(sen.split(' ')[:max_length])
        
        target = confusion_matrix_mapping[ label[9:] ]

        word_embeddings = torch.tensor(embed_model.get_word_vector(sen.split(' ')[0]), dtype = float).view(1, -1)

        for word in sen.split(' ')[1:]:
            word_embed = torch.tensor(embed_model.get_word_vector(word), dtype = float)
            word_embeddings = torch.vstack((word_embeddings, word_embed))

        for _ in range( max_length - len(sen.split(' ')) ):
            word_embeddings = torch.vstack((word_embeddings, torch.zeros(embed_model_dim, dtype = float) ))

        # print(word_embeddings.shape)
        # print(torch.transpose(word_embeddings, 0, 1).shape)

        return torch.transpose(word_embeddings, 0, 1), target

class SentenceEmbeding():
    def __call__(self, sample, embed_model_dim, embed_model):
        sen, label = sample

        sen_embed = torch.tensor([ 0 for _ in range(embed_model_dim) ], dtype = torch.float)

        target = confusion_matrix_mapping[ label[9:] ]

        for word in sen.split(' '):
            sen_embed += torch.tensor(embed_model.get_word_vector(word), dtype = float)
        
        # sen_embed /= len(sen.split(' '))
        
        return sen_embed, target


def get_dataloaders(X_train, X_test, X_valid, y_train, y_test, y_valid, batch_size):
    
    train = DATA(X_train, y_train, input_dimension, embed_model, max_length, transform=WordEmbeding())
    test = DATA(X_test, y_test, input_dimension, embed_model, max_length, transform=WordEmbeding())
    valid = DATA(X_valid, y_valid, input_dimension, embed_model, max_length, transform=WordEmbeding())


    train_dl = torch.utils.data.DataLoader(train,
                                                batch_size=batch_size,
                                                shuffle=True
                                            )

    test_dl = torch.utils.data.DataLoader(test,
                                                batch_size=len(y_test),
                                                shuffle=False
                                            )

    valid_dl = torch.utils.data.DataLoader(valid,
                                                batch_size=batch_size,
                                                shuffle=False
                                            )

    return train_dl, test_dl, valid_dl

train_dataloader, test_dataloader, valid_dataloader = get_dataloaders(train_X, test_X, valid_X, train_y, test_y, valid_y, batch_size = 256)




# training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# nueral network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.randn((input_dimension, max_length), dtype = torch.float, device = device)
        
        self.softmax = nn.Softmax()

        self.fc1 = nn.Linear(input_dimension, 64)
        # nn.init.xavier_uniform_(self.fc1.weight)
        
        # self.fc2 = nn.Linear(64, 128)
        # nn.init.xavier_uniform_(self.fc1.weight)

        self.lrelu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        
        self.dropout = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(64, 20)
        # nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # print(x.shape)
        
        x = torch.mul(x, self.attn)
        x = torch.sum(x, dim=2)
        # x = self.softmax(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.sigmoid(x)
        # x = self.fc2(x)
        # x = self.sigmoid(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x





net = Net()
net.to(device)
net.attn.to(device)
net.softmax.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


print('start training')
train_acc, valid_acc = [], []
i=0



for epoch in range(1):  # loop over the dataset multiple times 
    running_loss_train = 0.0
    running_loss_valid = 0.0
    print(epoch)
    for data_train in train_dataloader:
        i+=1
        
        inputs_train, labels_train = data_train[0].to(device), data_train[1].to(device)
        optimizer.zero_grad()
        
        # print(inputs_train.get_device())
        # print(net.attn.get_device())

        # net.attn = net.softmax(net.attn)

        # print(net.attn.shape)
        # print(inputs_train.shape)
        # x = torch.matmul(inputs_train, net.attn)
        
        # print(x.shape)
        inputs_train = inputs_train.float()
        outputs_train = net(inputs_train)

        # print(outputs_train.shape)

        loss_train = criterion(outputs_train, labels_train)
        loss_train.backward()
        
        optimizer.step()
        scheduler.step()

        running_loss_train += loss_train.item()

        if i % 100 == 0:    # print every 2000 mini-batches
            correct = 0
            total = 0
            with torch.no_grad():
                for data_valid in valid_dataloader:
                    inputs_valid, labels_valid = data_valid[0].to(device), data_valid[1].to(device)
                    
                    # print(inputs_valid.shape)
                    # print(net.attn.shape)
                    
                    # x_valid = torch.matmul(inputs_valid, net.attn)
                    
                    # x_valid = x_valid.float()

                    inputs_valid = inputs_valid.float()

                    outputs_valid = net(inputs_valid)

                    loss_valid = criterion(outputs_valid, labels_valid)

                    running_loss_valid += loss_valid.item()

                    _, predicted = torch.max(outputs_valid.data, 1)
            
                    total += labels_valid.size(0)
                    correct += (predicted == labels_valid).sum().item()

            print(f'Accuracy of the network on the {total} valid inputs: {100 * correct / total} ')

        
        
            print(f'[{epoch + 1}, {i + 1:5d}] Training loss: {running_loss_train / 100:.3f}')
            print(f'[{epoch + 1}, {i + 1:5d}] Valid loss: {running_loss_valid}')
            running_loss_train = 0.0
            running_loss_valid = 0.0
    
    
    
    
    
    correct = 0
    total = 0
  
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data_valid in valid_dataloader:

            inputs_valid, labels_valid = data_valid[0].to(device), data_valid[1].to(device)
            
            inputs_valid = inputs_valid.float()
            
            outputs_valid = net(inputs_valid)
            # the class with the highest energy is what we choose as prediction
            
            # debug()
            _, predicted = torch.max(outputs_valid.data, 1)
            
            total += labels_valid.size(0)
            correct += (predicted == labels_valid).sum().item()
    valid_acc.append(100 * correct / total)
    print(f'Accuracy of the network on the {total} valid inputs: {100 * correct / total} ')

    
    
    
    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in train_dataloader:
    #         inputs, labels = data
    #         # calculate outputs by running inputs through the network
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = net(inputs)
    #         # the class with the highest energy is what we choose as prediction
            
    #         # debug()
    #         _, predicted = torch.max(outputs.data, 1)
            
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # train_acc.append(100 * correct / total)
    # print(f'Accuracy of the network on the {total} train inputs: {100 * correct / total} %')


print('Finished Training')
# print(max(train_acc), max(test_acc))



torch.save(net, '../result/basline_nn_simple.pt')

model_scripted = torch.jit.script(net) # Export to TorchScript
model_scripted.save('../result/basline_nn_jit.pt')
