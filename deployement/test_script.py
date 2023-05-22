from working.IndicLID import IndicLID

IndicLID_model = IndicLID()



f_test_native = open('corpus/test_combine_native.txt', 'r')
lines_test_native = f_test_native.read().split('\n')
f_test_native.close()

lines_test_native = [line for line in lines_test_native if line]

lines_test_native_x = [' '.join(line.split(' ')[1:]) for line in lines_test_native if line] 
lines_test_native_y = [line.split(' ')[0][9:] for line in lines_test_native if line] 

lang_mapping = {
            'Assamese' : 'asm_Beng',
            'Bangla' : 'ben_Beng',
            'Bodo' : 'brx_Deva',
            'Dogri' : 'doi_Deva',
            'Gujarati' : 'guj_Gujr',
            'Hindi' : 'hin_Deva',
            'Kannada' : 'kan_Knda',
            'Kashmiri_Arab' : 'kas_Arab',
            'Kashmiri_Deva' : 'kas_Deva',
            'Konkani' : 'kok_Deva',
            'Maithili' : 'mai_Deva',
            'Malayalam' : 'mal_Mlym',
            'Manipuri_Beng' : 'mni_Beng',
            'Manipuri_Mei' : 'mni_Meti',
            'Marathi' : 'mar_Deva',
            'Nepali' : 'nep_Deva',
            'Oriya' : 'ori_Orya',
            'Punjabi' : 'pan_Guru',
            'Sanskrit' : 'san_Deva',
            'Santali' : 'sat_Olch',
            'Sindhi' : 'snd_Arab',
            'Tamil' : 'tam_Tamil',
            'Telugu' : 'tel_Telu',
            'Urdu' : 'urd_Arab'
        }



lines_test_native_y = [ lang_mapping[language_label] for language_label in  lines_test_native_y ]

output_test_native = IndicLID_model.batch_predict(lines_test_native_x, 32)


pred_label = [output[1] for output in output_test_native]

count = 0
for gd, pred in zip(lines_test_native_y, pred_label):
    if gd == pred:
        count+=1

print(count/len(pred_label))









f_test_roman = open('corpus/test_combine_roman.txt', 'r')
lines_test_roman = f_test_roman.read().split('\n')
f_test_roman.close()

lines_test_roman_x = [' '.join(line.split(' ')[1:]) for line in lines_test_roman if line] 
lines_test_roman_y = [line.split(' ')[0][9:] for line in lines_test_roman if line] 

lang_mapping = {
            'Assamese' : 'asm_Latn',
            'Bangla' : 'ben_Latn',
            'Bodo' : 'brx_Latn',
            'Dogri' : 'doi_Latn',
            'Gujarati' : 'guj_Latn',
            'Hindi' : 'hin_Latn',
            'Kannada' : 'kan_Latn',
            'Kashmiri' : 'kas_Latn',
            'Konkani' : 'kok_Latn',
            'Maithili' : 'mai_Latn',
            'Malayalam' : 'mal_Latn',
            'Manipuri_Beng' : 'mni_Latn',
            'Manipuri_Mei' : 'mni_Latn',
            'Marathi' : 'mar_Latn',
            'Nepali' : 'nep_Latn',
            'Oriya' : 'ori_Latn',
            'Punjabi' : 'pan_Latn',
            'Sanskrit' : 'san_Latn',
            'Santali' : 'sat_Latn',
            'Sindhi' : 'snd_Latn',
            'Tamil' : 'tam_Latn',
            'Telugu' : 'tel_Latn',
            'Urdu' : 'urd_Latn'
        }



lines_test_roman_y = [ lang_mapping[language_label] for language_label in  lines_test_roman_y ]

output_test_roman = IndicLID_model.batch_predict(lines_test_roman_x, 32)

pred_label = [output[1] for output in output_test_roman]

count = 0
for gd, pred in zip(lines_test_roman_y, pred_label):
    if gd == pred:
        count+=1
        
print(count/len(pred_label))
