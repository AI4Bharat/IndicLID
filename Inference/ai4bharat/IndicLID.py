# import packages

import os
import sys
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
import random

import fasttext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer
import transformers



class IndicBERT_Data(Dataset):
    def __init__(self, indices, X):
        self.size = len(X)
        self.x = X
        self.i = indices

        # self.y = Y
        # self.transform = transform

    def __len__(self):
        return (self.size)

    def __getitem__(self, idx):
        # print(self.x)
        
        text = self.x[idx]
        # text = sample[0]

        index = self.i[idx]
        


        # if self.transform:
        #     sample = self.transform(sample)
        # target = self.IndicLID_lang_code_dict[ label[9:] ]
        
        return tuple([index, text])


class IndicLID():

    def __init__(self, input_threshold = 0.5, roman_lid_threshold = 0.6):
        # define dictionary for roman and native languages to langauge code
        # define input_threhsold percentage for native and roman script input diversion 
        # define model_threhsold for roman script model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.IndicLID_FTN_path = 'models/indiclid-ftn/model_baseline_roman.bin'
        self.IndicLID_FTR_path = 'models/indiclid-ftr/model_baseline_roman.bin'
        self.IndicLID_BERT_path = 'models/indiclid-bert/basline_nn_simple.pt'

        self.IndicLID_FTN = fasttext.load_model(self.IndicLID_FTN_path)
        self.IndicLID_FTR = fasttext.load_model(self.IndicLID_FTR_path)
        self.IndicLID_BERT = torch.load(self.IndicLID_BERT_path, map_location = self.device)
        self.IndicLID_BERT.eval()
        self.IndicLID_BERT_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
        
        self.input_threshold = input_threshold
        self.model_threshold = roman_lid_threshold
        self.classes = 47     
        
        self.IndicLID_lang_code_dict = {
            'asm_Latn' : 0,
            'ben_Latn' : 1,
            'brx_Latn' : 2,
            'guj_Latn' : 3,
            'hin_Latn' : 4,
            'kan_Latn' : 5,
            'kas_Latn' : 6,
            'kok_Latn' : 7,
            'mai_Latn' : 8,
            'mal_Latn' : 9,
            'mni_Latn' : 10,
            'mar_Latn' : 11,
            'nep_Latn' : 12,
            'ori_Latn' : 13,
            'pan_Latn' : 14,
            'san_Latn' : 15,
            'snd_Latn' : 16,
            'tam_Latn' : 17,
            'tel_Latn' : 18,
            'urd_Latn' : 19,
            'eng_Latn' : 20,
            'other' : 21,
            'asm_Beng' : 22,
            'ben_Beng' : 23,
            'brx_Deva' : 24,
            'doi_Deva' : 25,
            'guj_Gujr' : 26,
            'hin_Deva' : 27,
            'kan_Knda' : 28,
            'kas_Arab' : 29,
            'kas_Deva' : 30,
            'kok_Deva' : 31,
            'mai_Deva' : 32,
            'mal_Mlym' : 33,
            'mni_Beng' : 34,
            'mni_Meti' : 35,
            'mar_Deva' : 36,
            'nep_Deva' : 37,
            'ori_Orya' : 38,
            'pan_Guru' : 39,
            'san_Deva' : 40,
            'sat_Olch' : 41,
            'snd_Arab' : 42,
            'tam_Tamil' : 43,
            'tel_Telu' : 44,
            'urd_Arab' : 45
        }



        self.IndicLID_lang_code_dict_reverse = {
            0 : 'asm_Latn',
            1 : 'ben_Latn',
            2 : 'brx_Latn',
            3 : 'guj_Latn',
            4 : 'hin_Latn',
            5 : 'kan_Latn',
            6 : 'kas_Latn',
            7 : 'kok_Latn',
            8 : 'mai_Latn',
            9 : 'mal_Latn',
            10 : 'mni_Latn',
            11 : 'mar_Latn',
            12 : 'nep_Latn',
            13 : 'ori_Latn',
            14 : 'pan_Latn',
            15 : 'san_Latn',
            16 : 'snd_Latn',
            17 : 'tam_Latn',
            18 : 'tel_Latn',
            19 : 'urd_Latn',
            20 : 'eng_Latn',
            21 : 'other',
            22 : 'asm_Beng',
            23 : 'ben_Beng',
            24 : 'brx_Deva',
            25 : 'doi_Deva',
            26 : 'guj_Gujr',
            27 : 'hin_Deva',
            28 : 'kan_Knda',
            29 : 'kas_Arab',
            30 : 'kas_Deva',
            31 : 'kok_Deva',
            32 : 'mai_Deva',
            33 : 'mal_Mlym',
            34 : 'mni_Beng',
            35 : 'mni_Meti',
            36 : 'mar_Deva',
            37 : 'nep_Deva',
            38 : 'ori_Orya',
            39 : 'pan_Guru',
            40 : 'san_Deva',
            41 : 'sat_Olch',
            42 : 'snd_Arab',
            43 : 'tam_Tamil',
            44 : 'tel_Telu',
            45 : 'urd_Arab'
        }

    def pre_process(self, input):
        # pre-process the input in the same way as we pro-process the training sample
        return input


    def char_percent_check(self, input):
        # check whether input has input_threhsold of roman characters
        
        # count total number of characters in string
        input_len = len(list(input))

        # count special character spaces and newline in string
        special_char_pattern = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        special_char_matches = special_char_pattern.findall(input)
        special_chars = len(special_char_matches)
        
        spaces = len(re.findall('\s', input))
        newlines = len(re.findall('\n', input))

        # subtract total-special character counts
        total_chars = input_len - (special_chars + spaces + newlines)

        # count the number of english character and digit in string
        en_pattern = re.compile('[a-zA-Z0-9]')
        en_matches = en_pattern.findall(input)
        en_chars = len(en_matches)

        # calculate the percentage of english character in total number of characters
        if total_chars == 0:
            return 0
        return (en_chars/total_chars)



    def native_inference(self, input_list, output_dict):

        if not input_list:
            return output_dict
        
        # inference for fasttext native script model
        input_texts = [line[1] for line in input_list]
        IndicLID_FTN_predictions = self.IndicLID_FTN.predict(input_texts)
        
        # add result of input directly to output_dict
        for input, pred_label, pred_score in zip(input_list, IndicLID_FTN_predictions[0], IndicLID_FTN_predictions[1]):
            # print(pred_score)
            output_dict[input[0]] = (input[1], pred_label[0][9:], pred_score[0], 'IndicLID-FTN')

        return output_dict

    def roman_inference(self, input_list, output_dict, batch_size):

        if not input_list:
            return output_dict
        
        # 1st stage
        # inference for fasttext roman script model
        input_texts = [line[1] for line in input_list]
        IndicLID_FTR_predictions = self.IndicLID_FTR.predict(input_texts)
        
        IndicLID_BERT_inputs = []
        # add result of input directly to output_dict
        for input, pred_label, pred_score in zip(input_list, IndicLID_FTR_predictions[0], IndicLID_FTR_predictions[1]):
            if pred_score[0] > self.model_threshold:
                output_dict[input[0]] = (input[1], pred_label[0][9:], pred_score[0], 'IndicLID-FTR')
            else:
                IndicLID_BERT_inputs.append(input)
        
        # 2nd stage
        output_dict = self.IndicBERT_roman_inference(IndicLID_BERT_inputs, output_dict, batch_size)
        return output_dict

    
    def IndicBERT_roman_inference(self, IndicLID_BERT_inputs, output_dict, batch_size):
        # inference for IndicBERT roman script model

        if not IndicLID_BERT_inputs:
            return output_dict
        
        df = pd.DataFrame(IndicLID_BERT_inputs)
        dataloader = self.get_dataloaders(df.iloc[:,0], df.iloc[:,1], batch_size)


        with torch.no_grad():
            for data in dataloader:
                batch_indices = data[0]
                batch_inputs = data[1]

                word_embeddings = self.IndicLID_BERT_tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    
                word_embeddings = word_embeddings.to(self.device)
            
                batch_outputs = self.IndicLID_BERT(word_embeddings['input_ids'], 
                            token_type_ids=word_embeddings['token_type_ids'], 
                            attention_mask=word_embeddings['attention_mask']
                            )
                

                _, batch_predicted = torch.max(batch_outputs.logits, 1)
            
            
                for index, input, pred_label, logit in zip(batch_indices, batch_inputs, batch_predicted, batch_outputs.logits):
                    output_dict[index] = (input,
                                            self.IndicLID_lang_code_dict_reverse[pred_label.item()], 
                                            logit[pred_label.item()].item(), 'IndicLID-BERT'
                                            )


        return output_dict


    def post_process(self, output_dict):
        # output the result in some consistent language code format
        results = []
        keys = list(output_dict.keys())
        keys.sort()
        for index in keys:
            results.append( output_dict[index] )

        return results
    
    def get_dataloaders(self, indices, input_texts, batch_size):
        data_obj = IndicBERT_Data(indices, input_texts)
        dl = torch.utils.data.DataLoader(data_obj,
                                                    batch_size=batch_size,
                                                    shuffle=False
                                                )
        return dl
        
    def predict(self, input):
        input_list = [input,]
        self.batch_predict(input_list, 1)

    def batch_predict(self, input_list, batch_size):

        # call functions seq by seq and divert the input to IndicBERT if 
        # fasttext prediction score is less than the defined model_threhsold.
        # Also output the inference time along with the result.
        output_dict = {}

        roman_inputs = []
        native_inputs = []

        # text roman percent check 
        for index, input in enumerate(input_list):
            if self.char_percent_check(input) > self.input_threshold:
                roman_inputs.append((index, input))
            else:
                native_inputs.append((index, input))
        
        output_dict = self.native_inference(native_inputs, output_dict)
        output_dict = self.roman_inference(roman_inputs, output_dict, batch_size)
        
        results = self.post_process(output_dict)
        return results
