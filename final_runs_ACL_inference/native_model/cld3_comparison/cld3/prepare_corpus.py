import random
from unittest.util import sorted_list_difference 
import csv
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
    'sat' : ['a', 'f'],
    'sd' : ['d', 'f'],
    'ta' : ['d', 'f'],
    'te' : ['d', 'f'],
    'ur' : ['a', 'd', 'f'],
    'ks_arab' : ['a', 'f'], 
    'ks_deva' : ['f'], 
    'mni_beng' : ['f'],
    'mni_mei' : ['a'],
    'en' : [],
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
    'en' : 'eng_Latn',
    'sat' : 'sat_Olck'
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
    'bn', 
    'gu', 'hi', 
    'kn', 'ml', 'mr', 
    'ne', 'pa', 'sd', 
    'ta', 'te', 'ur', 
    ]

all_lang_test_lines = []

stats = [['Language', 'Dakshina', 'Flores', 'AI4B', 'Total', 'Deduplicates']]

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    lines_duplicates = []
    Dakshina_count = 0
    Flores_count = 0
    AI4B_count = 0


    if 'd' in test_data_sources[lang_code]:
        test_file_name = '../../../../../Dakshina/scored_dakshina/final_merge_native_set/'+lang_code+'_filter.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()


        lines_in_test = [line for line in lines_in_test if line ]

        lines_duplicates += lines_in_test

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test
        
        Dakshina_count = len(lines_in_test)


    if 'f' in test_data_sources[lang_code]:
        test_file_name = '../../../../../flores200/flores200_dataset/devtest/'+flores_dict[lang_code]+'.devtest'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()
    


        lines_in_test = [line for line in lines_in_test if line ]

        lines_duplicates += lines_in_test

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test
        
        Flores_count = len(lines_in_test)

    if 'a' in test_data_sources[lang_code]:
        test_file_name = '../../../../../Other_sources/annotator_ai4bharat_train_test_split/test/'+lang_code + '_test.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()
    


        lines_in_test = [line for line in lines_in_test if line ]

        lines_duplicates += lines_in_test

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        AI4B_count = len(lines_in_test)
        
    stats.append([lang_code_dict[lang_code], Dakshina_count, Flores_count, AI4B_count, Dakshina_count + Flores_count + AI4B_count, len(set(lines_duplicates))])


with open('../corpus/lang_stat_test_native.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for line in stats:
        csvwriter.writerow(line)   

all_lang_test_lines = list(set(all_lang_test_lines))

random.shuffle(all_lang_test_lines)
f_out_test_rom = open('../corpus/test_combine_native_12.txt', 'w')
f_out_test_rom.write('\n'.join(all_lang_test_lines))
f_out_test_rom.close()










# # Dakshina
# lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'ta', 'te', 'ur', 'sd']

# all_lang_test_lines_nat = []

# for lang_code in lang_code_list:

#     print("lang : ", lang_code_dict[lang_code])

#     test_file_name = '../../../../../Dakshina/scored_dakshina/final_merge_native_set/'+lang_code+'_filter.txt'
#     f_in_test = open(test_file_name, 'r')

#     lines_in_test = f_in_test.read().split('\n')
#     f_in_test.close()

#     lines_in_test = [line for line in lines_in_test if line ]

#     lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

#     all_lang_test_lines_nat += lines_in_test
    

# all_lang_test_lines_nat = list(set(all_lang_test_lines_nat))

# random.shuffle(all_lang_test_lines_nat)
# f_out_test_rom = open('test_dakshina_native.txt', 'w')
# f_out_test_rom.write('\n'.join(all_lang_test_lines_nat))
# f_out_test_rom.close()





# # Flores 200
# lang_code_list = [
#     'as',
#     'bn', 
#     'gu',
#     'hi', 
#     'kn', 
#     'mai', 
#     'ml', 
#     'mr', 
#     'ne', 
#     'or', 
#     'pa', 
#     'sa', 
#     'sd', 
#     'ta', 
#     'te', 
#     'ks_arab',
#     'ks_deva', 
#     'mni_beng',
#     'ur',
#     'en'
#     ]
# all_lang_test_lines_nat = []

# for lang_code in lang_code_list:

#     print("lang : ", lang_code_dict[lang_code])

#     test_file_name = '../../../../../flores200/flores200_dataset/devtest/'+flores_dict[lang_code]+'.devtest'
#     f_in_test = open(test_file_name, 'r')

#     lines_in_test = f_in_test.read().split('\n')
#     f_in_test.close()

#     lines_in_test = [line for line in lines_in_test if line ]

#     lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

#     all_lang_test_lines_nat += lines_in_test
    

# all_lang_test_lines_nat = list(set(all_lang_test_lines_nat))

# random.shuffle(all_lang_test_lines_nat)
# f_out_test_rom = open('test_flores200_native.txt', 'w')
# f_out_test_rom.write('\n'.join(all_lang_test_lines_nat))
# f_out_test_rom.close()






# # AI4Bharat
# lang_code_list = [
#     'brx', 'dg',
#     'gom',  
#     'mai',  
#     'ne', 'sa', 'sat', 
#     'ur', 
#     'ks_arab', 'mni_mei'
#     ]

# all_lang_test_lines_nat = []

# for lang_code in lang_code_list:

#     print("lang : ", lang_code_dict[lang_code])

#     test_file_name = '../../../../../Other_sources/annotator_ai4bharat_train_test_split/test/'+lang_code + '_test.txt'
#     f_in_test = open(test_file_name, 'r')

#     lines_in_test = f_in_test.read().split('\n')
#     f_in_test.close()

#     lines_in_test = [line for line in lines_in_test if line ]

#     lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

#     all_lang_test_lines_nat += lines_in_test

# all_lang_test_lines_nat = list(set(all_lang_test_lines_nat))

# random.shuffle(all_lang_test_lines_nat)
# f_out_test_rom = open('test_AI4Bharat_native.txt', 'w')
# f_out_test_rom.write('\n'.join(all_lang_test_lines_nat))
# f_out_test_rom.close()

