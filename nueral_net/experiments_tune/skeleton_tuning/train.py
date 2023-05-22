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
import wandb
# import os

# os.environ["WANDB_API_KEY"] = '80e73c3565f964351f9de16f3538dae036768a7f'
# os.environ["WANDB_MODE"] = "offline"

max_length = 30
input_dimension = 512

# training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_embed_model(input_dimension, embed_model_lr):
    embed_model = fasttext.train_unsupervised(
            input = '../corpus/train_combine.txt',
            minn=3, 
            maxn=6, 
            dim=input_dimension,
            model='skipgram',
            lr=embed_model_lr
            )
    return embed_model

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
    def __init__(self, X, Y, embed_model_dim, embed_model, len_normalize, transform = None):
        self.size = len(Y)
        self.x = X
        self.y = Y
        self.transform = transform
        self.embed_model_dim = embed_model_dim
        self.embed_model = embed_model
        self.len_normalize = len_normalize

    def __len__(self):
        return (self.size)

    def __getitem__(self, idx):
        #print(self.x[idx].shape,self.y[idx].shape )
        sample = self.x[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample, self.embed_model_dim, self.embed_model, self.len_normalize)
        
        return sample


class SentenceEmbeding():
    def __call__(self, sample, embed_model_dim, embed_model, len_normalize):
        sen, label = sample

        sen_embed = torch.tensor([ 0 for _ in range(embed_model_dim) ], dtype = torch.float)

        target = confusion_matrix_mapping[ label[9:] ]

        for word in sen.split(' '):
            sen_embed += torch.tensor(embed_model.get_word_vector(word), dtype = float)
        
        if len_normalize:
            sen_embed /= len(sen.split(' '))
        
        return sen_embed, target


def get_dataloaders(X_train, X_test, X_valid, y_train, y_test, y_valid, config, embed_model):
    
    train = DATA(X_train, y_train, config.input_dimension, embed_model, config.len_normalize, transform=SentenceEmbeding())
    test = DATA(X_test, y_test, config.input_dimension, embed_model, config.len_normalize, transform=SentenceEmbeding())
    valid = DATA(X_valid, y_valid, config.input_dimension, embed_model, config.len_normalize, transform=SentenceEmbeding())


    train_dl = torch.utils.data.DataLoader(train,
                                                batch_size=config.batch_size,
                                                shuffle=True
                                            )

    test_dl = torch.utils.data.DataLoader(test,
                                                batch_size=config.batch_size,
                                                shuffle=False
                                            )

    valid_dl = torch.utils.data.DataLoader(valid,
                                                batch_size=config.batch_size,
                                                shuffle=False
                                            )

    return train_dl, test_dl, valid_dl


# nueral network
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.fc1 = nn.Linear(config.input_dimension, config.hidden_dimension)
        
        if config.weight_init == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
        

        self.fc2 = nn.Linear(config.hidden_dimension, 20)
        
        if config.weight_init == 'xavier':
            nn.init.xavier_uniform_(self.fc2.weight)

        # self.fc2 = nn.Linear(64, 128)
        # nn.init.xavier_uniform_(self.fc1.weight)

        if config.activation == 'lrelu':
            self.activation_fun = nn.LeakyReLU(0.1)
        elif config.activation == 'sigmoid':
            self.activation_fun = nn.Sigmoid()
        elif config.activation == 'gelu':
            self.activation_fun = nn.GELU()
        elif config.activation == 'tanh':
            self.activation_fun = nn.Tanh()
        
        if config.droput:
            self.dropout = nn.Dropout(p=config.dropout_value)



    def forward(self, x, config):

        x = self.fc1(x)

        if config.activation != 'linear':
            x = self.activation_fun(x)
        
        if config.droput:
            x = self.dropout(x)

        x = self.fc2(x)
        return x


def train(config):

    embed_model = train_embed_model(config.input_dimension, config.embed_model_lr)

    train_dataloader, test_dataloader, valid_dataloader = get_dataloaders(train_X, test_X, valid_X, train_y, test_y, valid_y, config, embed_model)
    
    
    net = Net(config)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print('start training')
    i=0
    for epoch in range(config.epochs):  # loop over the dataset multiple times 
        running_loss_train = 0.0
        running_loss_valid = 0.0

        for data_train in train_dataloader:
            i+=1            

            inputs_train, labels_train = data_train[0].to(device), data_train[1].to(device)

            optimizer.zero_grad()

            outputs_train = net(inputs_train, config)

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
                        
                        outputs_valid = net(inputs_valid, config)

                        loss_valid = criterion(outputs_valid, labels_valid)

                        running_loss_valid += loss_valid.item()

                        _, predicted = torch.max(outputs_valid.data, 1)
                
                        total += labels_valid.size(0)
                        correct += (predicted == labels_valid).sum().item()

                # print(f'Accuracy of the network on the {total} valid inputs: {100 * correct / total} ')
                val_acc = 100 * correct / total
                wandb.log({"val_accuracy": val_acc})

            
            
                # print(f'[{epoch + 1}, {i + 1:5d}] Training loss: {running_loss_train / 100:.3f}')
                wandb.log({"Training_loss": (running_loss_train / 100) })
                wandb.log({"Validation_loss": (running_loss_valid / 100) })

                # print(f'[{epoch + 1}, {i + 1:5d}] Valid loss: {running_loss_valid}')
                running_loss_train = 0.0
                running_loss_valid = 0.0
        
  
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data_train in train_dataloader:
                inputs_train, labels_train = data_train[0].to(device), data_train[1].to(device)
                outputs_train = net(inputs_train, config)
                _, predicted = torch.max(outputs_train.data, 1)
                
                total += labels_train.size(0)
                correct += (predicted == labels_train).sum().item()
        
        print(f'Accuracy of the network on the {total} train inputs: {100 * correct / total} %')


    print('Finished Training')


# sweep config
sweep_config = {
    'method' : 'bayes',
    'metric' : {
        'name' : 'val_accuracy',
        'goal' : 'maximize'
    },
    'parameters' : {
        'input_dimension' : {
            'values' : [16, 32, 64, 128, 256, 512, 768, 1024]
        },
        'embed_model_lr' : {
            'values' : [0.9, 0.7, 0.5, 0.3, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.009, 0.007, 0.005, 0.003, 0.001]
        },
        'len_normalize' : {
            'values' : [True, False]
        },
        'batch_size' : {
            'values' : [16, 32, 64, 128, 256]
        },
        'lr' : {
            'values' : [0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.009, 0.007, 0.005, 0.003, 0.001, 0.0009, 0.0007, 0.0005, 0.0003, 0.0001]
        },
        'weight_decay' : {
            'values' : [0.1, 0.001, 0.0001, 0.00001, 0.000001]
        },
        'hidden_dimension' : {
            'values' : [16, 32, 64, 128, 256, 512, 1024]
        }, 
        'weight_init' : {
            'values' : ['xavier', 'random']
        }, 
        'activation' : {
            'values' : ['lrelu', 'sigmoid', 'tanh', 'gelu']
        }, 
        'droput' : {
            'values' : [True, False]
        }, 
        'dropout_value' : {
            'values' : [0.1, 0.2, 0.3, 0.4, 0.5]
        }, 
        'epochs' : {
            'values' : [1, 2, 3, 4, 5]
        }
    }
}



def train_callable():
    config_defaults = {
        'input_dimension' : 512,
        'embed_model_lr' : 0.5,
        'len_normalize' : True,
        'batch_size' : 32,
        'lr' : 0.005,
        'weight_decay' : 0.0001,
        'hidden_dimension' : 64,
        'weight_init' : 'random',
        'activation' : 'sigmoid',
        'droput' : False,
        'dropout_value' : 1.0,
        'epochs' : 1
    }

    # wandb intialization
    wandb.init(config = config_defaults, project = "IndicLID", entity='cs20s002' )
    config = wandb.config

    # setting run name
    exp_name = "input_dimension" + str(config.input_dimension) + "_embed_model_lr_" + str(config.embed_model_lr) + "_len_normalize_" + str(config.len_normalize) + "_batch_size_" + str(config.batch_size) + "_lr_" + str(config.lr) + "_weight_decay_" + str(config.weight_decay) + "_hidden_dimension_"+ str(config.hidden_dimension) + "_weight_init_" + str(config.weight_init) + "_activation_" + str(config.activation) + "_droput_" + str(config.droput) + "_dropout_value_" + str(config.dropout_value)
    

    wandb.run.name = exp_name

    # building model
    train(config)    

sweep_id = wandb.sweep(sweep_config,  entity = "cs20s002", project = "IndicLID")
wandb.agent(sweep_id, train_callable, count = 200)
