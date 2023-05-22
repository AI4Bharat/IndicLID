import random
from unittest.util import sorted_list_difference 
train_data_sources = {
    'as' : ['i', 'w', 'v'],
    'bn' : ['i', 'w', 'v'],
    'brx' : ['i', 'n', 'a', 'v'],
    'dg' : ['i', 'v'],
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
    'sa' : ['i', 'n', 'a', 'w', 'v'],
    'sat' : ['i', 'n', 'w', 'v'],
    'sd' : ['i', 'w'],
    'ta' : ['i', 'w', 'v'],
    'te' : ['i', 'w', 'v'],
    'ur' : ['i', 'a'],
    'ks_arab' : ['n', 'a'], 
    'ks_deva' : ['n'], 
    'mni_beng' : ['n'],
    'mni_mei' : ['i', 'a', 'w'],
    'en' : ['i'],
    'other' : ['nws']
}
test_data_sources = {
    'as' : ['f'],
    'bn' : ['d', 'f'],
    'brx' : ['a'],
    'dg' : ['a'],
    'gom' : ['a'], 
    'gu' : ['d', 'f'],
    'hi' : ['d', 'f'],
    'kn' : ['d', 'f'],
    'mai' : ['a', 'f'],
    'ml' : ['d', 'f'],
    'mr' : ['d', 'f'],
    'ne' : ['a', 'f'],
    'or' : ['f'],
    'pa' : ['d', 'f'],
    'sa' : ['a', 'f'],
    'sat' : ['a'],
    'sd' : ['d', 'f'],
    'ta' : ['d', 'f'],
    'te' : ['d', 'f'],
    'ur' : ['a', 'd', 'f'],
    'ks_arab' : ['a', 'f'], 
    'ks_deva' : ['f'], 
    'mni_beng' : ['f'],
    'mni_mei' : ['a'],
    'en' : ['f'],
}

valid_data_sources = {
    'as' : ['f'],
    'bn' : ['d', 'f'],
    'brx' : ['a'],
    'dg' : ['a'],
    'gom' : ['a'], 
    'gu' : ['d', 'f'],
    'hi' : ['d', 'f'],
    'kn' : ['d', 'f'],
    'mai' : ['a', 'f'],
    'ml' : ['d', 'f'],
    'mr' : ['d', 'f'],
    'ne' : ['a', 'f'],
    'or' : ['f'],
    'pa' : ['d', 'f'],
    'sa' : ['a', 'f'],
    'sat' : ['a'],
    'sd' : ['d', 'f'],
    'ta' : ['d', 'f'],
    'te' : ['d', 'f'],
    'ur' : ['a', 'd', 'f'],
    'ks_arab' : ['a', 'f'], 
    'ks_deva' : ['f'], 
    'mni_beng' : ['f'],
    'mni_mei' : ['a'],
    'en' : ['f'],
}

flores_dict = {
    'as' : 'asm_Beng',
    'bn' : 'ben_Beng', 
    'gu' : 'guj_Gujr',
    'hi' : 'hin_Deva', 
    'kn' : 'kan_Knda', 
    'mai' : 'mai_Deva', 
    'ml' : 'mal_Mlym', 
    'mr' : 'mar_Deva', 
    'ne' : 'npi_Deva', 
    'or' : 'ory_Orya', 
    'pa' : 'pan_Guru', 
    'sa' : 'san_Deva', 
    'sd' : 'snd_Arab', 
    'ta' : 'tam_Taml', 
    'te' : 'tel_Telu', 
    'ks_arab' : 'kas_Arab', 
    'ks_deva' : 'kas_Deva',
    'mni_beng' : 'mni_Beng', 
    'ur' : 'urd_Arab',
    'en' : 'eng_Latn'
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
    'ks_arab' : 'Kashmiri_Arab', 
    'ks_deva' : 'Kashmiri_Deva', 
    'mni_beng' : 'Manipuri_Beng',
    'mni_mei' : 'Manipuri_Mei',
    'en' : 'English',
    'other' : 'Other'
}






# test data
lang_code_list = [
    'as', 'brx', 'bn', 
    'gom', 'gu', 'hi', 
    'kn', 'mai', 'ml', 'mr', 
    'ne', 'or', 'pa', 'sa', 'sd', 
    'ta', 'te', 'ur', 
    'ks_arab', 'ks_deva', 'mni_mei', 'mni_beng', 
    'dg', 'sat' , 'en'
    ]

all_lang_test_lines = []

file_lang = open('../corpus/lang_stat_test.txt', 'w')

file_lang.write('Testing Data\n')

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    total = 0

    if 'd' in test_data_sources[lang_code]:
        test_file_name = '../../../../../../../Dakshina/dakshina_dataset_v1.0/'+lang_code+'/romanized/'+lang_code+'.romanized.rejoined.test.native.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()


        lines_in_test = [line for line in lines_in_test if line ]

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Dakshina' + '\t' + str(len(lines_in_test)))
        file_lang.write('\n')
        
        total += len(lines_in_test)

    if 'f' in test_data_sources[lang_code]:
        test_file_name = '../../../../../../../flores200/flores200_dataset/devtest/'+flores_dict[lang_code]+'.devtest'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()
    


        lines_in_test = [line for line in lines_in_test if line ]

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Flores200' + '\t' + str(len(lines_in_test)))
        file_lang.write('\n')
        
        total += len(lines_in_test)

    if 'a' in test_data_sources[lang_code]:
        test_file_name = '../../../../../../../Other_sources/annotator_ai4bharat_train_test_split/test/'+lang_code + '_test.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()
    


        lines_in_test = [line for line in lines_in_test if line ]

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'AI4Bharat' + '\t' + str(len(lines_in_test)))
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









# valid data
lang_code_list = [
    'as', 'brx', 'bn', 
    'gom', 'gu', 'hi', 
    'kn', 'mai', 'ml', 'mr', 
    'ne', 'or', 'pa', 'sa', 'sd', 
    'ta', 'te', 'ur', 
    'ks_arab', 'ks_deva', 'mni_mei', 'mni_beng', 
    'dg', 'sat' ,'en'
    ]

all_lang_valid_lines = []

file_lang = open('../corpus/lang_stat_valid.txt', 'w')

file_lang.write('Validation Data\n')

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    total = 0

    if 'd' in valid_data_sources[lang_code]:
        valid_file_name = '../../../../../../../Dakshina/dakshina_dataset_v1.0/'+lang_code+'/romanized/'+lang_code+'.romanized.rejoined.dev.native.txt'
        f_in_valid = open(valid_file_name, 'r')

        lines_in_valid = f_in_valid.read().split('\n')
        f_in_valid.close()


        lines_in_valid = [line for line in lines_in_valid if line ]

        lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]

        all_lang_valid_lines += lines_in_valid

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Dakshina' + '\t' + str(len(lines_in_valid)))
        file_lang.write('\n')
        
        total += len(lines_in_valid)

    if 'f' in valid_data_sources[lang_code]:
        valid_file_name = '../../../../../../../flores200/flores200_dataset/dev/'+flores_dict[lang_code]+'.dev'
        f_in_valid = open(valid_file_name, 'r')

        lines_in_valid = f_in_valid.read().split('\n')
        f_in_valid.close()
    


        lines_in_valid = [line for line in lines_in_valid if line ]

        lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]

        all_lang_valid_lines += lines_in_valid

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'Flores200' + '\t' + str(len(lines_in_valid)))
        file_lang.write('\n')
        
        total += len(lines_in_valid)

    if 'a' in valid_data_sources[lang_code]:
        valid_file_name = '../../../../../../../Other_sources/annotator_ai4bharat_train_test_split/valid/'+lang_code + '_valid.txt'
        f_in_valid = open(valid_file_name, 'r')

        lines_in_valid = f_in_valid.read().split('\n')
        f_in_valid.close()
    


        lines_in_valid = [line for line in lines_in_valid if line ]

        lines_in_valid = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_valid ]

        all_lang_valid_lines += lines_in_valid

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'AI4Bharat' + '\t' + str(len(lines_in_valid)))
        file_lang.write('\n')
        
        total += len(lines_in_valid)
    
    file_lang.write(lang_code_dict[lang_code] + '\t' + 'Total' + '\t' + str(total))
    file_lang.write('\n')
    

    

file_lang.write('\n')
random.shuffle(all_lang_valid_lines)


f_out_valid_rom = open('../corpus/valid_combine.txt', 'w')

f_out_valid_rom.write('\n'.join(all_lang_valid_lines))

f_out_valid_rom.close()
file_lang.close()










all_lang_valid_lines_without_label = [' '.join(line.split(' ')[1:]) for line in all_lang_valid_lines]
all_lang_test_lines_without_label = [' '.join(line.split(' ')[1:]) for line in all_lang_test_lines]





# train data
lang_code_list = [
    'as', 'brx', 'bn', 
    'gom', 'gu', 'hi', 
    'kn', 'mai', 'ml', 'mr', 
    'ne', 'or', 'pa', 'sa', 'sd', 
    'ta', 'te', 'ur', 
    'ks_arab', 'ks_deva', 'mni_mei', 'mni_beng', 
    'dg', 'sat' , 'en', 'other'
    ]

samples = 100000
all_lang_train_lines = []


file_lang = open('../corpus/lang_stat_train.txt', 'w')

file_lang.write('Training Data\n')

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])


    lines_in_total = []

    if 'i' in train_data_sources[lang_code]:
        train_file_name = '../../../../../../../preprocess_indiccorp/IndicCorp_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
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
        train_file_name = '../../../../../../../Other_sources/nllb_preprocess/nllb_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
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
        train_file_name = '../../../../../../../preprocess_wikipedia/wikipedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
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

    if 'nws' in train_data_sources[lang_code]:
        train_file_name = '../../../../../../../preprocess_news_crawl/news_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        f_in_train = open(train_file_name, 'r')

        lines_in_train = f_in_train.read().split('\n')
        f_in_train.close()

        lines_in_train = lines_in_train[:samples]

        lines_in_train = [line for line in lines_in_train if line ]

        print("News : len of lines_in_train : ", len(lines_in_train))

        lines_in_train = list( set(lines_in_train).difference( set(all_lang_valid_lines_without_label) ) )
        lines_in_train = list( set(lines_in_train).difference( set(all_lang_test_lines_without_label) ) )

        print("News : len of lines_in_train after removing test and valid data : ", len(lines_in_train))

        lines_in_train = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_train ]

        file_lang.write(lang_code_dict[lang_code] + '\t' + 'News' + '\t' + str(len(lines_in_train)))
        file_lang.write('\n')  

        lines_in_total += lines_in_train

    if 'v' in train_data_sources[lang_code]:
        train_file_name = '../../../../../../../preprocess_vikaspedia/vikaspedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
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
        train_file_name = '../../../../../../../Other_sources/annotator_ai4bharat_prerocess/train_tok_norm_romanized_100k_sample_cleaned/'+lang_code+'/'+lang_code+'_indic_tok.txt'
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

file_lang.write('Test Data - Dakshina original native\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../../../../../../Dakshina/dakshina_dataset_v1.0/'+lang_code+'/romanized/'+lang_code+'.romanized.rejoined.test.native.txt'
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


f_out_test_rom = open('../corpus/test_dakshina_original_native.txt', 'w')

f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))

f_out_test_rom.close()



















lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'ta', 'te', 'ur', 'sd']

all_lang_valid_lines_rom = []

file_lang.write('valid_dakshina_original_native\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    valid_file_name = '../../../../../../../Dakshina/dakshina_dataset_v1.0/'+lang_code+'/romanized/'+lang_code+'.romanized.rejoined.dev.native.txt'
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


f_out_valid_rom = open('../corpus/valid_dakshina_original_native.txt', 'w')

f_out_valid_rom.write('\n'.join(all_lang_valid_lines_rom))

f_out_valid_rom.close()

















# Flores 200
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
    'mni_beng',
    'ur',
    'en'
    ]
all_lang_test_lines_rom = []

file_lang.write('Test Data - Flores200\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../../../../../../flores200/flores200_dataset/devtest/'+flores_dict[lang_code]+'.devtest'
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


f_out_test_rom = open('../corpus/test_combine_flores200.txt', 'w')

f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))

f_out_test_rom.close()








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
    'mni_beng',
    'ur',
    'en'
    ]

all_lang_valid_lines_rom = []

file_lang.write('Validation Data - Flores200\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    valid_file_name = '../../../../../../../flores200/flores200_dataset/dev/'+flores_dict[lang_code]+'.dev'
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


f_out_valid_rom = open('../corpus/valid_combine_flores200.txt', 'w')

f_out_valid_rom.write('\n'.join(all_lang_valid_lines_rom))

f_out_valid_rom.close()










# AI4Bharat
lang_code_list = [
    'brx', 'dg',
    'gom',  
    'mai',  
    'ne', 'sa', 'sat', 
    'ur', 
    'ks_arab', 'mni_mei'
    ]

all_lang_test_lines_rom = []

file_lang.write('Test Data - AI4Bharat\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../../../../../../Other_sources/annotator_ai4bharat_train_test_split/test/'+lang_code + '_test.txt'
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


f_out_test_rom = open('../corpus/test_combine_AI4Bharat.txt', 'w')

f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))

f_out_test_rom.close()






lang_code_list = [
    'brx', 'dg',
    'gom',  
    'mai',  
    'ne', 'sa', 'sat', 
    'ur', 
    'ks_arab', 'mni_mei'
    ]



all_lang_valid_lines_rom = []

file_lang.write('Validation Data - AI4Bharat\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    valid_file_name = '../../../../../../../Other_sources/annotator_ai4bharat_train_test_split/valid/'+lang_code + '_valid.txt'
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


f_out_valid_rom = open('../corpus/valid_combine_AI4Bharat.txt', 'w')

f_out_valid_rom.write('\n'.join(all_lang_valid_lines_rom))

f_out_valid_rom.close()
file_lang.close()
