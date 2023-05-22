import zipfile
import shutil
import os 
import csv
import sys

sys_arg = sys.argv[1]
# extract
# lang_code_list = ['as', 'bd', 'bn', 'dg', 'en', 'gom', 'gu', 'hi', 'kha', 'kn', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'san', 'sat', 'sd', 'ta', 'te', 'ur']
lang_code_list = [sys_arg,]
file_stats = open('indiccorp_combine_stats.csv', 'a')
writer = csv.writer(file_stats)

# header = ['lang_code', 'Unique', 'Wiki' ,'Vikas', 'Total', 'Total_Unique']
# writer.writerow(header)

for lang_code in lang_code_list:
    
    print(lang_code)
    reading_dir = '../IndicCorp/'+lang_code
    
    output_dir = 'IndicCorp_data'
    # os.mkdir(output_dir + '/' + lang_code)
    output_dir = output_dir + '/' + lang_code

    # os.mkdir(output_dir + '/extracted_files')
    
    # extract unique sentence file
    # unique_sent_zip_file = reading_dir+'/'+lang_code+'_sents.zip'
    # zipfile.ZipFile(unique_sent_zip_file, 'r').extractall(output_dir+'/extracted_files')
    
    # extract wiki sentence file
    # wiki_sent_zip_file = reading_dir+'/wiki_feb2022_sents_'+lang_code+'.zip'
    # zipfile.ZipFile(wiki_sent_zip_file, 'r').extractall(output_dir+'/extracted_files')
    
    # combine
    records = []
    records.append(lang_code)
    unique_sent_extracted_file = output_dir+'/extracted_files/'+lang_code+'_sent.txt'
    wiki_sent_extracted_file = output_dir+'/extracted_files/outputs/'+lang_code+'.txt'
    vikas_sent_extracted_file = reading_dir+'/vikaspedia_sents_'+lang_code+'.txt'

    file_unique = open(unique_sent_extracted_file, 'r')
    lines_unique = file_unique.read().split('\n')
    file_unique.close()
    records.append(len(lines_unique))
    # records.append(0)
    print('Unique: ',len(lines_unique))

    file_wiki = open(wiki_sent_extracted_file, 'r')
    lines_wiki = file_wiki.read().split('\n')
    file_wiki.close()
    records.append(len(lines_wiki))
    # records.append(0)
    print('Wiki: ', len(lines_wiki))

    file_vikas = open(vikas_sent_extracted_file, 'r')
    lines_vikas = file_vikas.read().split('\n')
    file_vikas.close()  
    records.append(len(lines_vikas))
    # records.append(0)
    print('Vikas: ', len(lines_vikas))
    
    combined_file_name = output_dir + '/' +lang_code + '_combine.txt'
    file_out = open(combined_file_name, 'w')

    lines_out = lines_unique  + lines_wiki + lines_vikas
    #  + lines_vikas
    # lines_unique  + lines_wiki + lines_vikas
    records.append(len(lines_out))
    print('Total: ', len(lines_out))
    
    lines_out = list(set(lines_out))
    records.append(len(lines_out))
    print('Total Unique: ',len(lines_out))

    file_out.write('\n'.join(lines_out))

    file_out.close()
    writer.writerow(records)

file_stats.close()