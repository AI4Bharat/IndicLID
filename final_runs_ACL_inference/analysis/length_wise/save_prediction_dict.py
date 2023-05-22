import fasttext
import json
import random
fasttext_lid_model_path='/nlsasfs/home/ai4bharat/yashkm/yash/indic-lid/final_runs/roman_model/fasttext/clean_samples/tune_run/basline_en_other/result_model_dim_8/model_baseline_roman.bin'
fasttext_model = fasttext.load_model(fasttext_lid_model_path)

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
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


IndicBERT_lid_model_path = '/nlsasfs/home/ai4bharat/yashkm/yash/indic-lid/final_runs/roman_model/finetuning/clean_samples/IndicBERT/unfreeze_layers/result_unfreeze_1/basline_nn_simple.pt'
model = torch.load(IndicBERT_lid_model_path)
model.eval()

threshold = 0.6

# tokenizer = AutoTokenizer.from_pretrained("")
# tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")

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


def get_dataloaders(X_test, y_test, batch_size):
    

    test = DATA(X_test, y_test)


    test_dl = torch.utils.data.DataLoader(test,
                                                batch_size=batch_size,
                                                shuffle=False
                                            )

    return test_dl


# file = open('../result/result_acc.txt', 'w')

def evaluate(file_name, model):
    classes = 22

    dict_predictions = {}

    df_test = convert_to_dataframe('../corpus/'+file_name+'.txt')

    test_X = df_test.iloc[:, 1]
    test_y = df_test.iloc[:, 0]
    
    test_dataloader = get_dataloaders(test_X, test_y, batch_size = 64)

    correct = 0
    total = 0
    final_count = 0
    final_n = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            
            inputs, labels = data[0], data[1]
            
            # print(inputs)
            # print(labels)

            word_embeddings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    
            word_embeddings = word_embeddings.to(device)
            labels = labels.to(device)

            outputs = model(word_embeddings['input_ids'], 
                        token_type_ids=word_embeddings['token_type_ids'], 
                        attention_mask=word_embeddings['attention_mask'], 
                        labels=labels)
            
            # loss = outputs.loss
            # loss =  criterion(outputs['logits'], labels)

            # running_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            
            # print(outputs.logits)
            # print(predicted)
            
            for sen, label, pred_label, logit in zip(inputs, labels, predicted, outputs.logits):
                
                fasttext_prediction = fasttext_model.predict(sen)
                fasttext_pred_label = fasttext_prediction[0][0]
                fasttext_pred_score = fasttext_prediction[1][0]
                
                pred_score = logit[pred_label.item()].item()
                
                if fasttext_pred_score > threshold:
                    if confusion_matrix_mapping[fasttext_pred_label[9:]] == label.item():
                        sen_len = len(sen.split(' '))
                        if sen_len in dict_predictions:
                            dict_predictions[sen_len].append((sen, 1)) 
                        else:
                            dict_predictions[sen_len] = [(sen, 1)]
                            
                    else:
                        sen_len = len(sen.split(' '))
                        if sen_len in dict_predictions:
                            dict_predictions[sen_len].append((sen, 0)) 
                        else:
                            dict_predictions[sen_len] = [(sen, 0)]

                else:
                    if label.item() == pred_label.item():
                        sen_len = len(sen.split(' '))
                        if sen_len in dict_predictions:
                            dict_predictions[sen_len].append((sen, 1)) 
                        else:
                            dict_predictions[sen_len] = [(sen, 1)]
                            
                    else:
                        sen_len = len(sen.split(' '))
                        if sen_len in dict_predictions:
                            dict_predictions[sen_len].append((sen, 0)) 
                        else:
                            dict_predictions[sen_len] = [(sen, 0)]                



    with open("../result/dict_predictions.json", "w") as outfile:
        json.dump(dict_predictions, outfile)
        
# evaluate( 'test_combine', model)
# evaluate( 'valid_combine', model)
# evaluate( 'valid_train_set_distribution', model)
# evaluate( 'test_combine_dakshina', model)
# evaluate( 'valid_combine_dakshina', model)
# evaluate( 'test_combine_flores200_romanized', model)
# # evaluate( '../../../../Dakshina/dakshina_filtered/test_combine_dakshina.txt', model)
# evaluate( 'test_combine_ai4b_romanized', model)
# # evaluate( '../corpus/train_combine.txt', model)


# evaluate( 'test_combine', model)
# evaluate( 'test_dakshina_original_roman', model)
# evaluate( 'valid_dakshina_original_roman', model)
evaluate( 'test_combine_roman', model)
# evaluate( 'test_dakshina_indicxlit_romanized', model)
# evaluate( 'test_combine_flores200_romanized', model)
# evaluate( 'test_combine_ai4b_romanized', model)


# file.close()

