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

unfreeze_layer = sys.argv[1]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load('../result_unfreeze_'+str(unfreeze_layer)+'/basline_nn_simple.pt')
model.eval()

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")

# data loading
confusion_matrix_mapping  = {
    'Assamese' : 0,
    'Bangla' : 1,
    'Bodo' : 2,
    'Dogri' : 3,
    'Konkani' : 4, 
    'Gujarati' : 5,
    'Hindi' : 6,
    'Kannada' : 7,
    'Kashmiri_Arab' : 8,
    'Kashmiri_Deva' : 9,
    'Maithili' : 10,
    'Malayalam' : 11,
    'Manipuri_Beng' : 12,
    'Manipuri_Mei' : 13,
    'Marathi' : 14,
    'Nepali' : 15,
    'Oriya' : 16,
    'Punjabi' : 17,
    'Sanskrit' : 18,
    'Santali' : 19,
    'Sindhi' : 20,
    'Tamil' : 21,
    'Telugu' : 22,
    'Urdu' : 23,
    'English' : 24,
    'Other' : 25
}

confusion_matrix_reverse_mapping  = {
    0 : 'Assamese',
    1 : 'Bangla',
    2 : 'Bodo',
    3 : 'Dogri',
    4 : 'Konkani', 
    5 : 'Gujarati',
    6 : 'Hindi',
    7 : 'Kannada',
    8 : 'Kashmiri_Arab',
    9 : 'Kashmiri_Deva',
    10 : 'Maithili',
    11 : 'Malayalam',
    12 : 'Manipuri_Beng',
    13 : 'Manipuri_Mei',
    14 : 'Marathi',
    15 : 'Nepali',
    16 : 'Oriya',
    17 : 'Punjabi',
    18 : 'Sanskrit',
    19 : 'Santali',
    20 : 'Sindhi',
    21 : 'Tamil',
    22 : 'Telugu',
    23 : 'Urdu',
    24 : 'English',
    25 : 'Other'
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


file = open('../result_unfreeze_'+str(unfreeze_layer)+'/result_acc.txt', 'w')

def evaluate(file_name, model):
    
    classes = 26
    
    # save confusion matrix
    confusion_matrix = []
    for i in range(classes):
        confusion_matrix.append( [0]*classes )

    # save predictions
    file_predictions = open('../result_unfreeze_'+str(unfreeze_layer)+'/predictions_'+file_name+'.csv', 'w')
    csv_writer_predictions = csv.writer(file_predictions)
    csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )

    file_predictions_right = open('../result_unfreeze_'+str(unfreeze_layer)+'/right_predictions_'+file_name+'.csv', 'w')
    csv_writer_predictions_right = csv.writer(file_predictions_right)
    csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


    file_predictions_wrong = open('../result_unfreeze_'+str(unfreeze_layer)+'/wrong_predictions_'+file_name+'.csv', 'w')
    csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
    csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



    df_test = convert_to_dataframe('../corpus/'+file_name+'.txt')

    test_X = df_test.iloc[:, 1]
    test_y = df_test.iloc[:, 0]
    
    test_dataloader = get_dataloaders(test_X, test_y, batch_size = 64)

    correct = 0
    total = 0
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
                pred_score = logit[pred_label.item()].item()
                csv_writer_predictions.writerow( [ sen, confusion_matrix_reverse_mapping[label.item()], confusion_matrix_reverse_mapping[pred_label.item()], pred_score ] )
                
                if label.item()==pred_label.item():
                    csv_writer_predictions_right.writerow( [ sen, confusion_matrix_reverse_mapping[label.item()], confusion_matrix_reverse_mapping[pred_label.item()], pred_score ] )
                else:
                    csv_writer_predictions_wrong.writerow( [ sen, confusion_matrix_reverse_mapping[label.item()], confusion_matrix_reverse_mapping[pred_label.item()], pred_score ] )

                confusion_matrix[ label.item() ][ pred_label.item() ] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = ((100 * correct) / total)
    print(f'Accuracy of the network on the {total} test inputs: {acc} ')
   
    file.write('Test Set ('+file_name+') : ' + str(acc) + '\n')
    file_predictions.close()
    file_predictions_right.close()
    file_predictions_wrong.close()




    # Computing precision, recall and f1
    precsison_recall_f1 = []
    for i in range(classes):
        precsison_recall_f1.append([0] * 3)

    precision_denominator = 0
    recall_denominator = 0
    f1_denominator = 0

    for i in range(classes):
        no_of_correctly_predicted = confusion_matrix[i][i]
        total_predictions_as_i = 0

        precision = 0
        recall = 0
        f1_value = 0

        # true predicted i values out of all predicted i values
        for j in range(classes):
            total_predictions_as_i += confusion_matrix[j][i]
        if (total_predictions_as_i != 0):
            precision = no_of_correctly_predicted/total_predictions_as_i
            precision_denominator += 1

        # true predicted i values out of all actual i values
        total_actual_values_of_i = sum(confusion_matrix[i])
        if (total_actual_values_of_i != 0):
            recall = no_of_correctly_predicted/total_actual_values_of_i
            recall_denominator += 1
        
        # f1 score
        if (precision + recall != 0):
            f1_value = (2 * precision * recall) / (precision + recall)
            f1_denominator += 1
            
        precsison_recall_f1[i][0] = precision
        precsison_recall_f1[i][1] = recall
        precsison_recall_f1[i][2] = f1_value

    avg_precision = sum([precsison_recall_f1[i][0] for i in range(classes)]) / precision_denominator
    avg_recall = sum([precsison_recall_f1[i][1] for i in range(classes)]) / recall_denominator
    avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(classes)]) / f1_denominator



    # to save confusion matrix and precision recall matrix
    
    for i in range(classes):
        precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

    precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
    precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )


    file_precision_recall_f1 = open('../result_unfreeze_'+str(unfreeze_layer)+'/precision_recall_f1_'+file_name+'.csv', 'w')
    precision_recall_f1_csv_writer = csv.writer(file_precision_recall_f1)

    for i in range(classes+2):
        precision_recall_f1_csv_writer.writerow(precsison_recall_f1[i])
    file_precision_recall_f1.close()




    for i in range(classes):
        confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 

    confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(classes)] )
    
    file_confusion_matrix = open('../result_unfreeze_'+str(unfreeze_layer)+'/confusion_matrix_'+file_name+'.csv', 'w')
    confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

    for i in range(classes+1):
        confusion_matrix_csv_writer.writerow(confusion_matrix[i])

    file_confusion_matrix.close()


evaluate('test_combine', model)
evaluate('test_dakshina_original_native', model)
evaluate('valid_dakshina_original_native', model)
evaluate('test_combine_flores200', model)
evaluate('valid_combine_flores200', model)
evaluate('test_combine_AI4Bharat', model)
evaluate('valid_combine_AI4Bharat', model)


file.close()

