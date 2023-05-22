import pandas as pd

native_data_dict = {
    'as' : [],
    'brx' : [],
    'gom' : [],
    'ks' : [],
    'mai' : [],
    'mni' : [],
    'ne' : [],
    'or' : [],
    'sa' : [],
}

roman_data_dict = {
    'as' : [],
    'brx' : [],
    'gom' : [],
    'ks' : [],
    'mai' : [],
    'mni' : [],
    'ne' : [],
    'or' : [],
    'sa' : [],
}

# load from benchmark_reports_pilot_1
lang_code_list = ['as', 'brx', 'gom', 'ks', 'mai', 'mni', 'ne', 'or', 'sa']

for lang_code in lang_code_list:
    sheet = pd.read_csv('../benchmark_reports_pilot_1/'+lang_code+'.csv')
    
    sheet = sheet[['input_text', 'output_text']]

    sheet = sheet.values.tolist()

    for line in sheet:
        if line[0] not in native_data_dict[lang_code]:
            native_data_dict[lang_code].append(line[0])
            roman_data_dict[lang_code].append(line[1])

# load from extra_pilot_1
xlsx_file_name = '../extra_pilot_1/Extra_sens_pilot_1_combined_final.xlsx'

sheet_names = ['as', 'brx', 'ks', 'or', 'mai', 'sa']

for sheet_name in sheet_names:
    sheet = pd.ExcelFile(xlsx_file_name).parse(sheet_name)
    # print(sheet)

    sheet = sheet.values.tolist()

    for line in sheet:
        if line[0] not in native_data_dict[lang_code]:
            native_data_dict[sheet_name].append(line[0])
            roman_data_dict[sheet_name].append(line[1])

for lang_code in native_data_dict:
    print('lang : ', lang_code)
    print('native : ', len(native_data_dict[lang_code][:512]))
    print('roman : ', len(roman_data_dict[lang_code][:512]))
    

for lang_code in native_data_dict:
    file = open('native_script/'+lang_code+'_native.txt', 'w')
    file.write('\n'.join(native_data_dict[lang_code][:512]))
    file.close()

    file = open('roman_script/'+lang_code+'_roman.txt', 'w')
    file.write('\n'.join(roman_data_dict[lang_code][:512]))
    file.close()
 