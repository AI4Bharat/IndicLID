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
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import random
from transformers import AutoModel, AutoTokenizer
import transformers
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# IndicBERT-MLM-only

classes = 22

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/IndicBERTv2-MLM-only", num_labels=classes)

model.to(device) 

# freeze layers
for layer in model.bert.encoder.layer:
    for param in layer.parameters():
        param.requires_grad = False

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
    'Urdu' : 19,
    'English' : 20,
    'Other' : 21
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
    19 : 'Urdu',
    20 : 'English',
    21 : 'Other'
}

def convert_to_dataframe(file_name):
    file_in = open(file_name, 'r')
    lines_in = file_in.read().split('\n')
    file_in.close()
    
    print(lines_in[0])

    lines_in = [line.split(' ') for line in lines_in if line]
    lines_in = [ [ line[0], ' '.join(line[1:]) ] for line in lines_in]
    
    print(lines_in[0])

    df = pd.DataFrame(lines_in)

    print(df)

    return df

df_train = convert_to_dataframe('../corpus/train_combine.txt')
df_test = convert_to_dataframe('../corpus/test_combine.txt')
df_valid = convert_to_dataframe('../corpus/valid_combine.txt')

print(df_train.shape)
train_X = df_train.iloc[:, 1]
train_y = df_train.iloc[:, 0]

test_X = df_test.iloc[:, 1]
test_y = df_test.iloc[:, 0]

valid_X = df_valid.iloc[:, 1]
valid_y = df_valid.iloc[:, 0]


class DATA(Dataset):
    def __init__(self, X, Y):
        self.size = len(Y)
        self.x = X
        self.y = Y
        # self.transform = transform        

    def __len__(self):
        return (self.size)

    def __getitem__(self, idx):

        sample = self.x[idx], self.y[idx]
        
        text, label = sample[0], sample[1]

        # if self.transform:
        #     sample = self.transform(sample)
        target = confusion_matrix_mapping[ label[9:] ]
        
        return tuple([text, target])


def get_dataloaders(X_train, X_test, X_valid, y_train, y_test, y_valid, batch_size):
    
    train = DATA(X_train, y_train)
    test = DATA(X_test, y_test)
    valid = DATA(X_valid, y_valid)


    train_dl = torch.utils.data.DataLoader(train,
                                                batch_size=batch_size,
                                                shuffle=True
                                            )

    test_dl = torch.utils.data.DataLoader(test,
                                                batch_size=batch_size,
                                                shuffle=False
                                            )

    valid_dl = torch.utils.data.DataLoader(valid,
                                                batch_size=batch_size,
                                                shuffle=False
                                            )

    return train_dl, test_dl, valid_dl

train_dataloader, test_dataloader, valid_dataloader = get_dataloaders(train_X, test_X, valid_X, train_y, test_y, valid_y, batch_size = 64)

# training

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00003, weight_decay=1e-5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)
scheduler = transformers.get_linear_schedule_with_warmup( optimizer, num_warmup_steps=1000, num_training_steps=20000 )


print('start training')
train_acc, valid_acc = [], []

i=0
epochs = 1
max_val_acc = 0.0

for epoch in range(epochs):  # loop over the dataset multiple times 
    running_loss_train = 0.0
    running_loss_valid = 0.0
    print(epoch)
    for data_train in train_dataloader:
        i+=1
        
        inputs_train, labels_train = data_train[0], data_train[1]
        
        optimizer.zero_grad()

        word_embeddings_train = tokenizer(inputs_train, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # print(word_embeddings_train.keys())
        
        labels_train = labels_train.to(device)
        word_embeddings_train = word_embeddings_train.to(device)

        outputs_train = model(word_embeddings_train['input_ids'], 
                             token_type_ids=word_embeddings_train['token_type_ids'], 
                             attention_mask=word_embeddings_train['attention_mask'], 
                             labels=labels_train)
        
        # outputs_train = model(**word_embeddings_train)

        # print(outputs_train)
        # labels_train
        # logits = outputs_train['logits']
        # logits = logits.to(device)

        loss_train = outputs_train.loss
        
        # loss_train = criterion(logits, labels_train)

        loss_train.backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        running_loss_train += loss_train.item()

        if i % 100 == 0:    # print every 2000 mini-batches
            correct = 0
            total = 0
            with torch.no_grad():
                for data_valid in valid_dataloader:
                    inputs_valid, labels_valid = data_valid[0], data_valid[1]

                    word_embeddings_valid = tokenizer(inputs_valid, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    
                    word_embeddings_valid = word_embeddings_valid.to(device)
                    labels_valid = labels_valid.to(device)

                    outputs_valid = model(word_embeddings_valid['input_ids'], 
                             token_type_ids=word_embeddings_valid['token_type_ids'], 
                             attention_mask=word_embeddings_valid['attention_mask'], 
                             labels=labels_valid)
                    
                    loss_valid = outputs_valid.loss
                    # loss_valid =  criterion(outputs_valid['logits'], labels_valid)

                    running_loss_valid += loss_valid.item()

                    _, predicted = torch.max(outputs_valid.logits, 1)
            
                    total += labels_valid.size(0)
                    correct += (predicted == labels_valid).sum().item()

                curr_val_acc = ((100 * correct) / total)
                if curr_val_acc > max_val_acc:
                    if max_val_acc:
                        os.remove('../result/basline_nn_simple.pt')
                    torch.save(model, '../result/basline_nn_simple.pt')

                    # model_scripted = torch.jit.script(model) # Export to TorchScript
                    # model_scripted.save('../result/basline_nn_jit.pt')
                    max_val_acc = curr_val_acc
                        
            # print(f'Accuracy of the network on the {total} valid inputs: {100 * correct / total} ')
            file_log = open('../result/acc_log.txt', 'a')
            file_log.write(f'Accuracy of the network on the {total} valid inputs: {100 * correct / total} '+'\n')

        
        
            # print(f'[{epoch + 1}, {i + 1:5d}] Training loss: {running_loss_train / 100:.3f}')
            # print(f'[{epoch + 1}, {i + 1:5d}] Valid loss: {running_loss_valid}')
            file_log.write(f'[{epoch + 1}, {i + 1:5d}] Training loss: {running_loss_train / 100:.3f}'+'\n')
            file_log.write(f'[{epoch + 1}, {i + 1:5d}] Valid loss: {running_loss_valid}'+'\n')
            file_log.close()
            
            running_loss_train = 0.0
            running_loss_valid = 0.0

            

    
    
    
    
    # correct = 0
    # total = 0
  
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data_valid in valid_dataloader:

    #         inputs_valid, labels_valid = data_valid[0].to(device), data_valid[1].to(device)
            
    #         word_embeddings_valid = tokenizer(inputs_valid, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
    #         word_embeddings_valid.to(device)
    #         labels_valid.to(device)
            
    #         outputs_valid = model(**word_embeddings_valid)
    #         # the class with the highest energy is what we choose as prediction
            
    #         # debug()
    #         _, predicted = torch.max(outputs_valid.data, 1)
            
    #         total += labels_valid.size(0)
    #         correct += (predicted == labels_valid).sum().item()
    # valid_acc.append(100 * correct / total)
    # print(f'Accuracy of the network on the {total} valid inputs: {100 * correct / total} ')

    
    
    
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



# torch.save(model, '../result/basline_nn_simple.pt')

# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('../result/basline_nn_jit.pt')
