import random
from unittest.util import sorted_list_difference 
train_data_sources = {
    'as' : ['i', 'w', 'v'],
    'bn' : ['i', 'w', 'v'],
    'brx' : ['i', 'n', 'a', 'v'],
    'gom' : ['n', 'a', 'w', 'v'], 
    'gu' : ['i', 'w', 'v'],
    'hi' : ['i', 'w', 'v'],
    'kn' : ['i', 'w', 'v'],
    'mai' : ['i', 'a', 'w', 'v'],
    'ml' : ['i', 'w', 'v'],
    'mr' : ['i', 'w', 'v'],
    'ne' : ['i', 'a', 'w', 'v'],
    'or' : ['i', 'w', 'v'],
    'pa' : ['i', 'w', 'v'],
    'sa' : ['n', 'a', 'w', 'v'],
    'sd' : ['i', 'w'],
    'ta' : ['i', 'w', 'v'],
    'te' : ['i', 'w', 'v'],
    'ur' : ['i', 'a'],
    'ks_arab' : ['n', 'a'], 
    'ks_deva' : ['n'], 
    'mni_mei' : ['i', 'a', 'w']
}
test_data_sources = {
    'bn' : ['d', ],
    'gu' : ['d', ],
    'hi' : ['d', ],
    'kn' : ['d', ],
    'ml' : ['d', ],
    'mr' : ['d', ],
    'pa' : ['d', ],
    'sd' : ['d', ],
    'ta' : ['d', ],
    'te' : ['d', ],
    'ur' : [ 'd', ],
}

valid_data_sources = {
    'as' : ['i', 'w', 'v'],
    'bn' : ['i', 'w', 'v', 'd'],
    'brx' : ['i', 'n', 'a', 'v'],
    'gom' : ['n', 'a', 'w', 'v'], 
    'gu' : ['i', 'w', 'v', 'd'],
    'hi' : ['i', 'w', 'v', 'd'],
    'kn' : ['i', 'w', 'v', 'd'],
    'mai' : ['i', 'a', 'w', 'v'],
    'ml' : ['i', 'w', 'v', 'd'],
    'mr' : ['i', 'w', 'v', 'd'],
    'ne' : ['i', 'a', 'w', 'v'],
    'or' : ['i', 'w', 'v'],
    'pa' : ['i', 'w', 'v', 'd'],
    'sa' : [ 'n', 'a', 'w', 'v'],
    'sd' : ['i', 'w', 'd'],
    'ta' : ['i', 'w', 'v', 'd'],
    'te' : ['i', 'w', 'v', 'd'],
    'ur' : ['i', 'a', 'd'],
    'ks_arab' : ['n', 'a'], 
    'ks_deva' : ['n'], 
    'mni_mei' : ['i', 'a', 'w']
}


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
    'mni_mei' : 'Manipuri_Mei'
}






# test data
lang_code_list = [
    'bn', 
    'gu', 'hi', 
    'kn', 'ml', 'mr', 
    'pa', 'sd', 
    'ta', 'te', 'ur', 
    ]

all_lang_test_lines = []

file_lang = open('../corpus/lang_stat_test.txt', 'w')

file_lang.write('Testing Data\n')

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    total = 0

    if 'd' in test_data_sources[lang_code]:
        test_file_name = '../../../../Dakshina/dakshina_dataset_v1.0/'+lang_code+'/romanized/'+lang_code+'.romanized.rejoined.test.roman.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()


        lines_in_test = [line for line in lines_in_test if line ]

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Dakshina' + '\t' + str(len(lines_in_test)))
        file_lang.write('\n')
        
        total += len(lines_in_test)
    
    file_lang.write(lang_code_dict[lang_code] + '\t' + 'Total' + '\t' + str(total))
    file_lang.write('\n')
    

    

file_lang.write('\n')
random.shuffle(all_lang_test_lines)


f_out_test_rom = open('../corpus/test_combine.txt', 'w')

f_out_test_rom.write('\n'.join(all_lang_test_lines))

f_out_test_rom.close()
file_lang.close()

all_lang_test_lines_without_label = [' '.join(line.split(' ')[1:]) for line in all_lang_test_lines]
























# valid data
lang_code_list = [
    'as', 'brx', 'bn', 
    'gom', 'gu', 'hi', 
    'kn', 'mai', 'ml', 'mr', 
    'ne', 'or', 'pa', 'sa', 'sd', 
    'ta', 'te', 'ur', 
    'ks_arab', 'ks_deva', 'mni_mei',
    ]

all_lang_valid_lines = []

# only valid set without dakshina
all_lang_valid_lines_without_dakshina = []


valid_samples = 5000

file_lang = open('../corpus/lang_stat_valid.txt', 'w')

file_lang.write('Validation Data\n')

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])



    lines_in_total = []

    if 'd' in valid_data_sources[lang_code]:
        valid_file_name = '../../../../Dakshina/dakshina_dataset_v1.0/'+lang_code+'/romanized/'+lang_code+'.romanized.rejoined.dev.roman.txt'
        f_in_valid = open(valid_file_name, 'r')

        lines_in_valid = f_in_valid.read().split('\n')
        f_in_valid.close()


        lines_in_valid = [line for line in lines_in_valid if line ]

        print("Dakshina : len of lines_in_valid : ", len(lines_in_valid))

        lines_in_valid = list( set(lines_in_valid).difference( set(all_lang_test_lines_without_label) ) )


        print("Dakshina : len of lines_in_valid after removing test and valid data : ", len(lines_in_valid))

        lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]

        lines_in_total += lines_in_valid

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Dakshina' + '\t' + str(len(lines_in_valid)))
        file_lang.write('\n')
        
        

    if 'i' in valid_data_sources[lang_code]:
        valid_file_name = '../../../../preprocess_indiccorp/IndicCorp_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_romanized.txt'
        f_in_valid = open(valid_file_name, 'r')

        lines_in_valid = f_in_valid.read().split('\n')
        f_in_valid.close()
    
        lines_in_valid = lines_in_valid[:valid_samples]

        lines_in_valid = [line for line in lines_in_valid if line ]
        
        print("IndicCorp : len of lines_in_valid : ", len(lines_in_valid))

        lines_in_valid = list( set(lines_in_valid).difference( set(all_lang_test_lines_without_label) ) )


        print("IndicCorp : len of lines_in_valid after removing test and valid data : ", len(lines_in_valid))
        
        lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'IndicCorp' + '\t' + str(len(lines_in_valid)))
        file_lang.write('\n')
        
        lines_in_total += lines_in_valid

        all_lang_valid_lines_without_dakshina += lines_in_valid

    if 'n' in valid_data_sources[lang_code]:
        valid_file_name = '../../../../Other_sources/nllb_preprocess/nllb_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_romanized.txt'
        f_in_valid = open(valid_file_name, 'r')

        lines_in_valid = f_in_valid.read().split('\n')
        f_in_valid.close()

        lines_in_valid = lines_in_valid[:valid_samples]

        lines_in_valid = [line for line in lines_in_valid if line ]

        print("NLLB : len of lines_in_valid : ", len(lines_in_valid))


        lines_in_valid = list( set(lines_in_valid).difference( set(all_lang_test_lines_without_label) ) )

        print("NLLB : len of lines_in_valid after removing test and valid data : ", len(lines_in_valid))

        lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]


        file_lang.write(lang_code_dict[lang_code] + '\t' + 'NLLB' + '\t' + str(len(lines_in_valid)))
        file_lang.write('\n')

        lines_in_total += lines_in_valid

        all_lang_valid_lines_without_dakshina += lines_in_valid

    if 'w' in valid_data_sources[lang_code]:
        valid_file_name = '../../../../preprocess_wikipedia/wikipedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_romanized.txt'
        f_in_valid = open(valid_file_name, 'r')

        lines_in_valid = f_in_valid.read().split('\n')
        f_in_valid.close()

        lines_in_valid = lines_in_valid[:valid_samples]

        lines_in_valid = [line for line in lines_in_valid if line ]

        print("Wikidata : len of lines_in_valid : ", len(lines_in_valid))

        lines_in_valid = list( set(lines_in_valid).difference( set(all_lang_test_lines_without_label) ) )

        print("Wikidata : len of lines_in_valid after removing test and valid data : ", len(lines_in_valid))

        lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Wikipedia' + '\t' + str(len(lines_in_valid)))
        file_lang.write('\n')  

        lines_in_total += lines_in_valid

        all_lang_valid_lines_without_dakshina += lines_in_valid

    if 'v' in valid_data_sources[lang_code]:
        valid_file_name = '../../../../preprocess_vikaspedia/vikaspedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_romanized.txt'
        f_in_valid = open(valid_file_name, 'r')

        lines_in_valid = f_in_valid.read().split('\n')
        f_in_valid.close()

        lines_in_valid = lines_in_valid[:valid_samples]

        lines_in_valid = [line for line in lines_in_valid if line ]

        print("Vikaspedia : len of lines_in_valid : ", len(lines_in_valid))

        lines_in_valid = list( set(lines_in_valid).difference( set(all_lang_test_lines_without_label) ) )

        print("Vikaspedia : len of lines_in_valid after removing test and valid data : ", len(lines_in_valid))

        lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Vikaspedia' + '\t' + str(len(lines_in_valid)))
        file_lang.write('\n')  

        lines_in_total += lines_in_valid

        all_lang_valid_lines_without_dakshina += lines_in_valid

    # if 'a' in valid_data_sources[lang_code]:
    #     valid_file_name = '../../../../Other_sources/annotator_ai4bharat_prerocess/train_tok_norm_romanized_100k_sample_cleaned/'+lang_code+'/'+lang_code+'_romanized.txt'
    #     f_in_valid = open(valid_file_name, 'r')

    #     lines_in_valid = f_in_valid.read().split('\n')
    #     f_in_valid.close()

    #     lines_in_valid = [line for line in lines_in_valid if line ]

    #     print("AI4B : len of lines_in_valid : ", len(lines_in_valid))


    #     lines_in_valid = list( set(lines_in_valid).difference( set(all_lang_test_lines_without_label) ) )

    #     print("AI4B : len of lines_in_valid after removing test and valid data : ", len(lines_in_valid))

    #     lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]


    #     file_lang.write(lang_code_dict[lang_code] + '\t' + 'AI4Bharat' + '\t' + str(len(lines_in_valid)))
    #     file_lang.write('\n')  

    #     lines_in_total += lines_in_valid

    lines_in_total = list(set(lines_in_total))
    random.shuffle(lines_in_total)

    all_lang_valid_lines += lines_in_total
     
    file_lang.write(lang_code_dict[lang_code] + '\t' + 'Total' + '\t' + str(len(lines_in_total)))
    file_lang.write('\n')
    

file_lang.write('\n')
random.shuffle(all_lang_valid_lines)


f_out_valid_rom = open('../corpus/valid_combine.txt', 'w')

f_out_valid_rom.write('\n'.join(all_lang_valid_lines))

f_out_valid_rom.close()
file_lang.close()


all_lang_valid_lines_without_label = [' '.join(line.split(' ')[1:]) for line in all_lang_valid_lines]






all_lang_valid_lines_without_dakshina = list(set(all_lang_valid_lines_without_dakshina))
random.shuffle(all_lang_valid_lines_without_dakshina)

f_out_valid_rom = open('../corpus/valid_train_set_distribution.txt', 'w')

f_out_valid_rom.write('\n'.join(all_lang_valid_lines_without_dakshina))

f_out_valid_rom.close()


























# train data
lang_code_list = [
    'as', 'brx', 'bn', 
    'gom', 'gu', 'hi', 
    'kn', 'mai', 'ml', 'mr', 
    'ne', 'or', 'pa', 'sa', 'sd', 
    'ta', 'te', 'ur', 
    'ks_arab', 'ks_deva', 'mni_mei',
    ]

samples = 100000
all_lang_train_lines = []


file_lang = open('../corpus/lang_stat_train.txt', 'w')

file_lang.write('Training Data\n')

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])


    lines_in_total = []

    if 'i' in train_data_sources[lang_code]:
        train_file_name = '../../../../preprocess_indiccorp/IndicCorp_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_romanized.txt'
        f_in_train = open(train_file_name, 'r')

        lines_in_train = f_in_train.read().split('\n')
        f_in_train.close()
    
        lines_in_train = lines_in_train[:samples]

        lines_in_train = [line for line in lines_in_train if line ]
        
        print("IndicCorp : len of lines_in_train : ", len(lines_in_train))

        lines_in_train = list( set(lines_in_train).difference( set(all_lang_valid_lines_without_label) ) )
        lines_in_train = list( set(lines_in_train).difference( set(all_lang_test_lines_without_label) ) )

        print("IndicCorp : len of lines_in_train after removing test and valid data : ", len(lines_in_train))
        
        lines_in_train = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_train ]

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'IndicCorp' + '\t' + str(len(lines_in_train)))
        file_lang.write('\n')
        
        lines_in_total += lines_in_train

    if 'n' in train_data_sources[lang_code]:
        train_file_name = '../../../../Other_sources/nllb_preprocess/nllb_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_romanized.txt'
        f_in_train = open(train_file_name, 'r')

        lines_in_train = f_in_train.read().split('\n')
        f_in_train.close()

        lines_in_train = lines_in_train[:samples]

        lines_in_train = [line for line in lines_in_train if line ]

        print("NLLB : len of lines_in_train : ", len(lines_in_train))

        lines_in_train = list( set(lines_in_train).difference( set(all_lang_valid_lines_without_label) ) )
        lines_in_train = list( set(lines_in_train).difference( set(all_lang_test_lines_without_label) ) )

        print("NLLB : len of lines_in_train after removing test and valid data : ", len(lines_in_train))

        lines_in_train = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_train ]


        file_lang.write(lang_code_dict[lang_code] + '\t' + 'NLLB' + '\t' + str(len(lines_in_train)))
        file_lang.write('\n')

        lines_in_total += lines_in_train


    if 'w' in train_data_sources[lang_code]:
        train_file_name = '../../../../preprocess_wikipedia/wikipedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_romanized.txt'
        f_in_train = open(train_file_name, 'r')

        lines_in_train = f_in_train.read().split('\n')
        f_in_train.close()

        lines_in_train = lines_in_train[:samples]

        lines_in_train = [line for line in lines_in_train if line ]

        print("Wikidata : len of lines_in_train : ", len(lines_in_train))

        lines_in_train = list( set(lines_in_train).difference( set(all_lang_valid_lines_without_label) ) )
        lines_in_train = list( set(lines_in_train).difference( set(all_lang_test_lines_without_label) ) )

        print("Wikidata : len of lines_in_train after removing test and valid data : ", len(lines_in_train))

        lines_in_train = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_train ]

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Wikipedia' + '\t' + str(len(lines_in_train)))
        file_lang.write('\n')  

        lines_in_total += lines_in_train

    if 'v' in train_data_sources[lang_code]:
        train_file_name = '../../../../preprocess_vikaspedia/vikaspedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_romanized.txt'
        f_in_train = open(train_file_name, 'r')

        lines_in_train = f_in_train.read().split('\n')
        f_in_train.close()

        lines_in_train = lines_in_train[:samples]

        lines_in_train = [line for line in lines_in_train if line ]

        print("Vikaspedia : len of lines_in_train : ", len(lines_in_train))

        lines_in_train = list( set(lines_in_train).difference( set(all_lang_valid_lines_without_label) ) )
        lines_in_train = list( set(lines_in_train).difference( set(all_lang_test_lines_without_label) ) )

        print("Vikaspedia : len of lines_in_train after removing test and valid data : ", len(lines_in_train))

        lines_in_train = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_train ]

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Vikaspedia' + '\t' + str(len(lines_in_train)))
        file_lang.write('\n')  

        lines_in_total += lines_in_train


    lines_in_total = list(set(lines_in_total))
    random.shuffle(lines_in_total)
    lines_in_total = lines_in_total[:samples]


    if 'a' in train_data_sources[lang_code]:
        train_file_name = '../../../../Other_sources/annotator_ai4bharat_prerocess/train_tok_norm_romanized_100k_sample_cleaned/'+lang_code+'/'+lang_code+'_romanized.txt'
        f_in_train = open(train_file_name, 'r')

        lines_in_train = f_in_train.read().split('\n')
        f_in_train.close()

        lines_in_train = [line for line in lines_in_train if line ]

        print("AI4B : len of lines_in_train : ", len(lines_in_train))

        lines_in_train = list( set(lines_in_train).difference( set(all_lang_valid_lines_without_label) ) )
        lines_in_train = list( set(lines_in_train).difference( set(all_lang_test_lines_without_label) ) )

        print("AI4B : len of lines_in_train after removing test and valid data : ", len(lines_in_train))

        lines_in_train = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_train ]


        file_lang.write(lang_code_dict[lang_code] + '\t' + 'AI4Bharat' + '\t' + str(len(lines_in_train)))
        file_lang.write('\n')  

        lines_in_total += lines_in_train

    lines_in_total = list(set(lines_in_total))
    
    file_lang.write(lang_code_dict[lang_code] + '\t' + 'Total' + '\t' + str(len(lines_in_total)))
    file_lang.write('\n')
    

    # resample logic
    if lang_code not in ['ks_arab', 'ks_deva']:
        if len(lines_in_total) < samples:
            lines_resampled = random.choices( lines_in_total, k=(samples - len(lines_in_total)) )
            lines_in_total += lines_resampled
            random.shuffle(lines_in_total)

    all_lang_train_lines += lines_in_total
    
    file_lang.write(lang_code_dict[lang_code] + '\t' + 'Resampled' + '\t' + str(len(lines_in_total)))
    file_lang.write('\n')

file_lang.write('\n')
random.shuffle(all_lang_train_lines)


f_out_train_rom = open('../corpus/train_combine.txt', 'w')

f_out_train_rom.write('\n'.join(all_lang_train_lines))

f_out_train_rom.close()
file_lang.close()
















































# separate test and valid files

file_lang = open('../corpus/lang_stat_test_valid.txt', 'w')

# Dakshina
lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'ta', 'te', 'ur', 'sd']

all_lang_test_lines_rom = []

file_lang.write('Test Data - Dakshina\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../../../Dakshina/dakshina_dataset_v1.0/'+lang_code+'/romanized/'+lang_code+'.romanized.rejoined.test.roman.txt'
    f_in_test = open(test_file_name, 'r')

    lines_in_test = f_in_test.read().split('\n')
    f_in_test.close()

    lines_in_test = [line for line in lines_in_test if line ]

    lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

    all_lang_test_lines_rom += lines_in_test
    
    file_lang.write(lang_code_dict[lang_code] + '\t' + str(len(lines_in_test)))
    file_lang.write('\n')

file_lang.write('\n')

random.shuffle(all_lang_test_lines_rom)


f_out_test_rom = open('../corpus/test_combine_dakshina.txt', 'w')

f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))

f_out_test_rom.close()



















lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'ta', 'te', 'ur', 'sd']

all_lang_valid_lines_rom = []

file_lang.write('Validation Data\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    valid_file_name = '../../../../Dakshina/dakshina_dataset_v1.0/'+lang_code+'/romanized/'+lang_code+'.romanized.rejoined.dev.roman.txt'
    f_in_valid = open(valid_file_name, 'r')

    lines_in_valid = f_in_valid.read().split('\n')
    f_in_valid.close()

    lines_in_valid = [line for line in lines_in_valid if line ]

    lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]

    all_lang_valid_lines_rom += lines_in_valid

    file_lang.write(lang_code_dict[lang_code] + '\t' + str(len(lines_in_valid)))
    file_lang.write('\n')
file_lang.write('\n')

random.shuffle(all_lang_valid_lines_rom)


f_out_valid_rom = open('../corpus/valid_combine_dakshina.txt', 'w')

f_out_valid_rom.write('\n'.join(all_lang_valid_lines_rom))

f_out_valid_rom.close()








# Flores_romanized
lang_code_list = [
    'as',
    'bn', 
    'gu',
    'hi', 
    'kn', 
    'mai', 
    'ml', 
    'mr', 
    'ne', 
    'or', 
    'pa', 
    'sa', 
    'sd', 
    'ta', 
    'te', 
    'ks_arab',
    'ks_deva',
    'ur'
    ]

all_lang_test_lines_rom = []

file_lang.write('Test Data - Flores_romanized\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../../../flores200/flores200_romanized/romanized_data/'+lang_code+'_romanized.txt'
    f_in_test = open(test_file_name, 'r')

    lines_in_test = f_in_test.read().split('\n')
    f_in_test.close()

    lines_in_test = [line for line in lines_in_test if line ]

    lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

    all_lang_test_lines_rom += lines_in_test
    
    file_lang.write(lang_code_dict[lang_code] + '\t' + str(len(lines_in_test)))
    file_lang.write('\n')

file_lang.write('\n')

random.shuffle(all_lang_test_lines_rom)


f_out_test_rom = open('../corpus/test_combine_flores200_romanized.txt', 'w')

f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))

f_out_test_rom.close()


file_lang.close()
