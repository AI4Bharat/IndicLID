import random

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

# Scored Dakshina romanized thorugh IndicXlit
lang_code_list = [
    'bn', 
    'gu', 'hi', 
    'kn', 'ml', 'mr', 
    'pa', 'sd', 
    'ta', 'te', 'ur', 
    ]

all_lang_test_lines_rom = []

# file_lang.write('Test Data - Scored Dakshina romanized thorugh IndicXlit\n')
for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../../../../Dakshina/scored_dakshina_romanized/romanized_data/'+lang_code+'_romanized.txt'
    f_in_test = open(test_file_name, 'r')

    lines_in_test = f_in_test.read().split('\n')
    f_in_test.close()

    lines_in_test = [line for line in lines_in_test if line ]

    lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

    all_lang_test_lines_rom += lines_in_test
    
    # file_lang.write(lang_code_dict[lang_code] + '\t' + str(len(lines_in_test)))
    # file_lang.write('\n')

# file_lang.write('\n')

random.shuffle(all_lang_test_lines_rom)


f_out_test_rom = open('../corpus/test_scored_dakshina_romanized.txt', 'w')

f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))

f_out_test_rom.close()

