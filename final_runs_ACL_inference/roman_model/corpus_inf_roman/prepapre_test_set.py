import random
import csv

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

test_data_sources = {
    'as' : ['b'],
    'bn' : ['d'],
    'brx' : ['b'],
    'gom' : ['b'], 
    'gu' : ['d'],
    'hi' : ['d'],
    'kn' : ['d'],
    'mai' : ['b'],
    'ml' : ['d'],
    'mr' : ['d'],
    'ne' : ['b'],
    'or' : ['b'],
    'pa' : ['d'],
    'sa' : ['b'],
    'sd' : ['d'],
    'ta' : ['d'],
    'te' : ['d'],
    'ur' : ['d'],
    'ks_arab' : ['b'], 
    'ks_deva' : [], 
    'mni_mei' : ['b'],
    'en' : [],
    'other' : []
}
 

# compile test_combine_roman (dakshina + Benchmark)
lang_code_list = [
    'as',
    'bn',
    'brx',
    'gom', 
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
    'ur',
    'ks_arab', 
    'mni_mei'
    ]

all_lang_test_lines = []

stats = [['Language', 'Dakshina', 'Benchmark', 'Total', 'Deduplicates']]

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    lines_duplicates = []
    Dakshina_count = 0
    Benchmark_count = 0
    if 'd' in test_data_sources[lang_code]:
        test_file_name = '../../Dakshina/scored_dakshina/final_merge_set/'+lang_code+'_filter.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()

        lines_in_test = [line for line in lines_in_test if line ]

        lines_duplicates += lines_in_test

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        Dakshina_count = len(lines_in_test)

    if 'b' in test_data_sources[lang_code]:
        test_file_name = '../../Benchmark/final_compiled_pilot_1/roman_script/'+lang_code+'_roman.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()

        lines_in_test = [line for line in lines_in_test if line ]

        lines_duplicates += lines_in_test

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        Benchmark_count = len(lines_in_test)

    stats.append([lang_code_dict[lang_code], Dakshina_count, Benchmark_count, Dakshina_count + Benchmark_count, len(set(lines_duplicates))])
    

with open('lang_stat_test_roman.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for line in stats:
        csvwriter.writerow(line)

all_lang_test_lines = list(set(all_lang_test_lines))

random.shuffle(all_lang_test_lines)
f_out_test_rom = open('test_combine_roman.txt', 'w')
f_out_test_rom.write('\n'.join(all_lang_test_lines))
f_out_test_rom.close()







# compile test_combine_romanized_indicxlit (dakshina + Benchmark)
lang_code_list = [
    'as',
    'bn',
    'brx',
    'gom', 
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
    'ur',
    'ks_arab', 
    'mni_mei'
    ]

all_lang_test_lines = []

stats = [['Language', 'Dakshina', 'Benchmark', 'Total', 'Deduplicates']]

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    lines_duplicates = []
    Dakshina_count = 0
    Benchmark_count = 0

    if 'd' in test_data_sources[lang_code]:
        test_file_name = '../../Dakshina/scored_dakshina_romanized/romanized_data/'+lang_code+'_romanized.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()

        lines_in_test = [line for line in lines_in_test if line ]

        lines_duplicates += lines_in_test

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        Dakshina_count = len(lines_in_test)

    if 'b' in test_data_sources[lang_code]:
        test_file_name = '../../Benchmark/final_compiled_pilot_1/romanized_indicxlit/romanized_data/'+lang_code+'_romanized.txt'
        f_in_test = open(test_file_name, 'r')

        lines_in_test = f_in_test.read().split('\n')
        f_in_test.close()

        lines_in_test = [line for line in lines_in_test if line ]

        lines_duplicates += lines_in_test

        lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

        all_lang_test_lines += lines_in_test

        Benchmark_count = len(lines_in_test)

    stats.append([lang_code_dict[lang_code], Dakshina_count, Benchmark_count, Dakshina_count + Benchmark_count, len(set(lines_duplicates))])
    

with open('lang_stat_test_romanized_indicxlit.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for line in stats:
        csvwriter.writerow(line)


all_lang_test_lines = list(set(all_lang_test_lines))

random.shuffle(all_lang_test_lines)
f_out_test_rom = open('test_combine_romanized_indicxlit.txt', 'w')
f_out_test_rom.write('\n'.join(all_lang_test_lines))
f_out_test_rom.close()


















# Dakshina filter roman test
lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'ta', 'te', 'ur', 'sd']

all_lang_test_lines_rom = []

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../Dakshina/scored_dakshina/final_merge_set/'+lang_code+'_filter.txt'
    f_in_test = open(test_file_name, 'r')

    lines_in_test = f_in_test.read().split('\n')
    f_in_test.close()

    lines_in_test = [line for line in lines_in_test if line ]

    lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

    all_lang_test_lines_rom += lines_in_test
    
all_lang_test_lines_rom = list(set(all_lang_test_lines_rom))
random.shuffle(all_lang_test_lines_rom)

f_out_test_rom = open('test_dakshina_roman.txt', 'w')
f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))
f_out_test_rom.close()




# Dakshina filter roman test romanized indicxlit
lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'ta', 'te', 'ur', 'sd']

all_lang_test_lines_rom = []

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../Dakshina/scored_dakshina_romanized/romanized_data/'+lang_code+'_romanized.txt'
    f_in_test = open(test_file_name, 'r')

    lines_in_test = f_in_test.read().split('\n')
    f_in_test.close()

    lines_in_test = [line for line in lines_in_test if line ]

    lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

    all_lang_test_lines_rom += lines_in_test
    
all_lang_test_lines_rom = list(set(all_lang_test_lines_rom))

random.shuffle(all_lang_test_lines_rom)

f_out_test_rom = open('test_dakshina_romanized_indicxlit.txt', 'w')
f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))
f_out_test_rom.close()











# Benchmark roman test
lang_code_list = [
    'as',
    'brx',
    'gom', 
    'mai',
    'ne',
    'or',
    'sa',
    'ks_arab', 
    'mni_mei'
    ]

all_lang_test_lines_rom = []

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../Benchmark/final_compiled_pilot_1/roman_script/'+lang_code+'_roman.txt'
    f_in_test = open(test_file_name, 'r')

    lines_in_test = f_in_test.read().split('\n')
    f_in_test.close()

    lines_in_test = [line for line in lines_in_test if line ]

    lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

    all_lang_test_lines_rom += lines_in_test
    
all_lang_test_lines_rom = list(set(all_lang_test_lines_rom))

random.shuffle(all_lang_test_lines_rom)

f_out_test_rom = open('test_benchmark_roman.txt', 'w')
f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))
f_out_test_rom.close()






# Benchmark filter roman test romanized indicxlit
lang_code_list = [
    'as',
    'brx',
    'gom', 
    'mai',
    'ne',
    'or',
    'sa',
    'ks_arab', 
    'mni_mei'
    ]

all_lang_test_lines_rom = []

for lang_code in lang_code_list:

    print("lang : ", lang_code_dict[lang_code])

    test_file_name = '../../Benchmark/final_compiled_pilot_1/romanized_indicxlit/romanized_data/'+lang_code+'_romanized.txt'
    f_in_test = open(test_file_name, 'r')

    lines_in_test = f_in_test.read().split('\n')
    f_in_test.close()

    lines_in_test = [line for line in lines_in_test if line ]

    lines_in_test = [ '__label__'+lang_code_dict[lang_code]+' '+line for line in lines_in_test ]

    all_lang_test_lines_rom += lines_in_test
    
all_lang_test_lines_rom = list(set(all_lang_test_lines_rom))

random.shuffle(all_lang_test_lines_rom)

f_out_test_rom = open('test_benchmark_romanized_indicxlit.txt', 'w')
f_out_test_rom.write('\n'.join(all_lang_test_lines_rom))
f_out_test_rom.close()
