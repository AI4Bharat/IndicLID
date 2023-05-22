import csv
import random

resource_dict = {
    'as' : ['u', 'v', 'w'],  
    'bd' : ['u', 'v'],  
    'bn' : ['u', 'v', 'w'],  
    'dg' : ['u', 'v'],  
    'en' : ['u'],  
    'gom' : ['u', 'v', 'w'],  
    'gu' : ['u', 'v', 'w'],  
    'hi' : ['u', 'v', 'w'],  
    'kha' : ['u'],  
    'kn' : ['u', 'v', 'w'],  
    'ks' : ['u', 'v', 'w'],  
    'mai' : ['u', 'v', 'w'],  
    'ml' : ['u', 'v', 'w'],  
    'mni' : [ 'w'],  
    'mr' : ['u', 'v', 'w'],  
    'ne' : ['u', 'v', 'w'],  
    'or' : ['u', 'v', 'w'],  
    'pa' : ['u', 'v', 'w'],  
    'sa' : ['u', 'v', 'w'],  
    'sat' : ['u', 'v', 'w'],  
    'sd' : ['u', 'w'],  
    'ta' : ['u', 'v', 'w'],  
    'te' : ['u', 'v', 'w'],  
    'ur' : ['u', 'v', 'w'],  
}

lang_code_list = ['as', 'bd', 'bn', 'dg', 'en', 'gom', 'gu', 'hi', 'kha', 'kn', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']
file_stats = open('indiccorp_combine_stats.csv', 'a')
writer = csv.writer(file_stats)

for lang_code in lang_code_list:
    
    records = []
    records.append(lang_code)
    unique_sent_extracted_file = '../../IndicCorp_data/'+lang_code+'/extracted_files/uniq_'+lang_code+'.txt'
    wiki_sent_extracted_file = '../../IndicCorp_data/'+lang_code+'/extracted_files/outputs/'+lang_code+'.txt'
    vikas_sent_extracted_file = '../../../IndicCorp/'+lang_code+'/vikaspedia_sents_'+lang_code+'.txt'

    lines_out = []
    if 'u' in resource_dict[lang_code]:
        file_unique = open(unique_sent_extracted_file, 'r')
        lines_unique = file_unique.read().split('\n')
        random.shuffle(lines_unique)
        lines_unique = lines_unique[:20000000] 
        file_unique.close()
        records.append(len(lines_unique))
        print('Unique: ',len(lines_unique))
        lines_out += lines_unique
    else:
        records.append(0)
        print('Unique: 0')

    if 'w' in resource_dict[lang_code]:

        file_wiki = open(wiki_sent_extracted_file, 'r')
        lines_wiki = file_wiki.read().split('\n')
        file_wiki.close()
        records.append(len(lines_wiki))
        print('Wiki: ', len(lines_wiki))
        lines_out += lines_wiki
    else:
        records.append(0)
        print('Wiki: 0')

    if 'v' in resource_dict[lang_code]:
        file_vikas = open(vikas_sent_extracted_file, 'r')
        lines_vikas = file_vikas.read().split('\n')
        file_vikas.close()  
        records.append(len(lines_vikas))
        print('Vikas: ', len(lines_vikas))
        lines_out += lines_vikas
    else:
        records.append(0)
        print('Vikas: 0')
    
    combined_file_name = '../' +lang_code + '/'+lang_code + '_combine_subset.txt'
    file_out = open(combined_file_name, 'w')


    records.append(len(lines_out))
    print('Total: ', len(lines_out))
    
    lines_out = list(set(lines_out))
    records.append(len(lines_out))
    print('Total Unique: ',len(lines_out))

    file_out.write('\n'.join(lines_out))

    file_out.close()
    writer.writerow(records)

file_stats.close()