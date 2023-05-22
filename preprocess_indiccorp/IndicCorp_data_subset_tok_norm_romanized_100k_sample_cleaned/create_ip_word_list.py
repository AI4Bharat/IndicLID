import re

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
lang_code_list = [
    'as', 'brx', 'bn', 
    'gom', 'gu', 'hi', 
    'kn', 'mai', 'ml', 'mr', 
    'ne', 'or', 'pa', 'sa', 'sd', 
    'ta', 'te', 'ur', 
    'mni_mei', 
    'dg', 'sat' 
    ]
# lang_code_list = ['ur']

for lang_code in lang_code_list:
    input_file_name = '../' +lang_code + '/'+lang_code + '_indic_tok.txt'
    file_in = open(input_file_name, 'r')
    lines_in = file_in.read().split('\n')

    word_list = []

    # tokenize word with mix language scripts e.g., 'भारतindia'
    # tokenize characters come along with numbers e.g., 'बोइंग737'
    final_pattern = lang_patterns_dict[lang_code_dict[lang_code]]
    for line in lines_in:
        word_list += re.findall(final_pattern, line)
    unique_word_list = list(set(word_list))
    
    unique_word_list = [ ' '.join(list(word)) for word in unique_word_list]

    output_file_name =  '../' + lang_code + '/' + lang_code + '_unique_word_list.txt'
    file_out = open(output_file_name, 'w')
    file_out.write('\n'.join(unique_word_list))
    file_out.close()
