import random
from unittest.util import sorted_list_difference 

lang_code_dict = {
    'as' : 'Assamese',
    'bn' : 'Bangla',
    'brx' : 'Bodo',
    'gom' : 'Konkani', 
    'gu' : 'Gujarati',
    'hi' : 'Hindi',
    'kn' : 'Kannada',
    'mai' : 'Maithili',
    'ml' : 'Malayalam',
    'mr' : 'Marathi',
    'ne' : 'Nepali',
    'or' : 'Oriya',
    'pa' : 'Punjabi',
    'sa' : 'Sanskrit',
    'sd' : 'Sindhi',
    'ta' : 'Tamil',
    'te' : 'Telugu',
    'ur' : 'Urdu',
    'dg' : 'Dogri',
    'sat' : 'Santali',
    'ks_arab' : 'Kashmiri', 
    'ks_deva' : 'Kashmiri', 
    'mni_beng' : 'Manipuri_Beng',
    'mni_mei' : 'Manipuri_Mei',
    'en' : 'English',
    'other' : 'Other'
}
reverse_lang_code_dict = {
    'Assamese' : 'as',
    'Bangla' : 'bn',
    'Bodo' : 'brx',
    'Konkani' : 'gom', 
    'Gujarati' : 'gu',
    'Hindi' : 'hi',
    'Kannada' : 'kn',
    'Maithili' : 'mai',
    'Malayalam' : 'ml',
    'Marathi' : 'mr',
    'Nepali' : 'ne',
    'Oriya' : 'or',
    'Punjabi' : 'pa',
    'Sanskrit' : 'sa',
    'Sindhi' : 'sd',
    'Tamil' : 'ta',
    'Telugu' : 'te',
    'Urdu' : 'ur',
    'Dogri' : 'dg',
    'Santali' : 'sat',
    'Kashmiri' : 'ks_arab', 
    'Kashmiri' : 'ks_deva', 
    'Manipuri_Beng' : 'mni_beng',
    'Manipuri_Mei' : 'mni_mei',
    'English' : 'en',
    'Other':  'other'
}
# train data
lang_code_list = [
    'as', 'brx', 'bn', 
    'gom', 'gu', 'hi', 
    'kn', 'mai', 'ml', 'mr', 
    'ne', 'or', 'pa', 'sa', 'sd', 
    'ta', 'te', 'ur', 
    'ks_arab', 'mni_mei', 'en', 'other'
    ]


file_in_train = open('../../../clean_corpus/train_combine.txt', 'r')
lines_in_train = file_in_train.read().split('\n')
file_in_train.close()

all_lang_train_lines = []

for line in lines_in_train:
    label = line.split(' ')[0]
    sen = ' '.join(line.split(' ')[1:])
    
    if reverse_lang_code_dict[label[9:]] in lang_code_list:
        all_lang_train_lines.append(line)


random.shuffle(all_lang_train_lines)


f_out_train = open('../corpus/train_combine.txt', 'w')

f_out_train.write('\n'.join(all_lang_train_lines))

f_out_train.close()






# valid data
file_in_valid = open('../../../clean_corpus/valid_combine.txt', 'r')
lines_in_valid = file_in_valid.read().split('\n')
file_in_valid.close()

all_lang_valid_lines = []

for line in lines_in_valid:
    label = line.split(' ')[0]
    sen = ' '.join(line.split(' ')[1:])
    
    if reverse_lang_code_dict[label[9:]] in lang_code_list:
        all_lang_valid_lines.append(line)


random.shuffle(all_lang_valid_lines)


f_out_valid = open('../corpus/valid_combine.txt', 'w')

f_out_valid.write('\n'.join(all_lang_valid_lines))

f_out_valid.close()








# test data
file_in_test = open('../../../clean_corpus/test_combine.txt', 'r')
lines_in_test = file_in_test.read().split('\n')
file_in_test.close()

all_lang_test_lines = []

for line in lines_in_test:
    label = line.split(' ')[0]
    sen = ' '.join(line.split(' ')[1:])
    
    if reverse_lang_code_dict[label[9:]] in lang_code_list:
        all_lang_test_lines.append(line)


random.shuffle(all_lang_test_lines)


f_out_test = open('../corpus/test_combine.txt', 'w')

f_out_test.write('\n'.join(all_lang_test_lines))

f_out_test.close()










# Dakshina filter roman test
lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'ta', 'te', 'ur', 'sd']

all_lang_test_lines_rom = []


for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../../../../Dakshina/scored_dakshina/final_merge_set/'+lang_code+'_filter.txt'
    f_in_test = open(test_file_name, 'r')

    lines_in_test = f_in_test.read().split('\n')
    f_in_test.close()

    lines_in_test = [line for line in lines_in_test if line ]

    lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

    all_lang_test_lines_rom += lines_in_test
    

random.shuffle(all_lang_test_lines_rom)


f_out_test_rom = open('../corpus/test_dakshina_filter_roman.txt', 'w')

f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))

f_out_test_rom.close()




