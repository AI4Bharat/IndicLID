import json
import re
import sys
from ai4bharat.transliteration import XlitEngine

e = XlitEngine( beam_width=4, rescore=False, src_script_type = "indic")

INDIC_TO_LATIN_PUNCT = {
    ## List of all punctuations across languages

    # Brahmic
    '।': '.', # Nagari
    ## Archaic Indic
    '॥': "..",  # Sanskrit
    '෴': '.', # Sinhala
    ## Meetei (influenced from Burmese)
    '꫰': ',',
    '꯫': '.',

    # Ol Chiki
    '᱾': '.',
    '᱿': '..',

    # Arabic
    '۔': '.',
    '؟': '?',
    '،': ',',
    '؛': ';',
    '۝': "..",
}

INDIC_TO_LATIN_PUNCT_TRANSLATOR = str.maketrans(INDIC_TO_LATIN_PUNCT)

lang_patterns_dict = {
    'Assamese' : "[\u0980-\u09FF]+",
    'Bangla' : "[\u0980-\u09FF]+",
    'Bodo' : "[\u0900-\u097F]+",
    'Konkani' : "[\u0900-\u097F]+", 
    'Gujarati' : "[\u0A80-\u0AFF]+",
    'Hindi' : "[\u0900-\u097F]+",
    'Kannada' : "[\u0C80-\u0CFF]+",
    'Maithili' : "[\u0900-\u097F]+",
    'Malayalam' : "[\u0D00-\u0D7F]+",
    'Marathi' : "[\u0900-\u097F]+",
    'Nepali' : "[\u0900-\u097F]+",
    'Oriya' : "[\u0B00-\u0B7F]+",
    'Punjabi' : "[\u0A00-\u0A7F]+",
    'Sanskrit' : "[\u0900-\u097F]+",
    'Sindhi' : "[\u0600-\u06FF]+",
    'Sinhala' : "[\u0D80-\u0DFF]+",
    'Tamil' : "[\u0B80-\u0BFF]+",
    'Telugu' : "[\u0C00-\u0C7F]+",
    'Urdu' : "[\u0600-\u06FF]+",
    'Dogri' : "[\u0900-\u097F]+",
    'Santali' : "[\u1C50-\u1C7F]+",
    'Kashmiri_Arab' : "[\u0600-\u089F]+",
    'Kashmiri_Deva' : "[\u0900-\u097F]+",
    'Manipuri_Mei' : "[\uABC0-\uABFF]+",
    'Manipuri_Beng' : "[\u0980-\u09FF]+",

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
    'mni_mei' : 'Manipuri_Mei'
}

# lang_code_list = [
#     'as', 'brx', 'bn', 
#     'gom', 'gu', 'hi', 
#     'kn', 'mai', 'ml', 'mni', 'mr', 
#     'ne', 'or', 'pa', 'sa', 'sd', 
#     'ta', 'te', 'ks', 'ur'
#     ]

lang_code = sys.argv[1]

# for lang_code in lang_code_list:

output_dict_file_name = '../'+lang_code+'/'+lang_code+'_output_dict.json'
output_dict = json.load( open(output_dict_file_name, 'r') )


input_file_name = '../' + lang_code + '/'+lang_code + '_indic_tok.txt'
file_in = open(input_file_name, 'r')
lines_in = file_in.read().split('\n')

pattern = re.compile(lang_patterns_dict[lang_code_dict[lang_code]])

lines_out = []
for line in lines_in:
    
    line = line.translate(INDIC_TO_LATIN_PUNCT_TRANSLATOR)
    
    temp_line = line.split(' ')
    
    # using output dictionary
    temp_line = [ output_dict[word] if word in output_dict else word for word in temp_line]
    
    # using api
    try:
        if lang_code == 'mni_mei':
            temp_line = [ e.translit_word(word, 'mni', topk=4)[0] if pattern.search(word) else word for word in temp_line]
        else:
            temp_line = [ e.translit_word(word, lang_code, topk=4)[0] if pattern.search(word) else word for word in temp_line]
    except:
        print('error in sentence')
    
    lines_out.append( ' '.join(temp_line) )

output_file_name = '../' + lang_code + '/'+lang_code + '_romanized.txt'
file_out = open(output_file_name, 'w')
file_out.write( '\n'.join(lines_out) )
file_out.close()
file_in.close()
