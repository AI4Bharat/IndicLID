import random


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


samples = 100

lang_code_list = ['brx', 'dg', 'gom', 'ks_arab', 'mai', 'mni_mei', 'ne', 'sa']


for lang_code in lang_code_list:

    lines_train_clean = []

    if 'i' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_indiccorp/IndicCorp_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'n' in train_data_sources[lang_code]:
        train_file_name = '../../Other_sources/nllb_preprocess/nllb_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'w' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_wikipedia/wikipedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'v' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_vikaspedia/vikaspedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'nws' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_news_crawl/news_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'a' in train_data_sources[lang_code]:
        train_file_name = '../../Other_sources/annotator_ai4bharat_prerocess/train_tok_norm_romanized_100k_sample_cleaned/'+lang_code+'/'+lang_code+'_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()


    lines_train_unclean = []

    if 'i' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_indiccorp/IndicCorp_data_subset_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'n' in train_data_sources[lang_code]:
        train_file_name = '../../Other_sources/nllb_preprocess/nllb_data_subset_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'w' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_wikipedia/wikipedia_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'v' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_vikaspedia/vikaspedia_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'nws' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_news_crawl/news_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'a' in train_data_sources[lang_code]:
        train_file_name = '../../Other_sources/annotator_ai4bharat_prerocess/train_tok_norm_romanized/'+lang_code+'/'+lang_code+'_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()




    file_in = open('../../Other_sources/annotator_ai4bharat_train_test_split/test/'+lang_code+'_test.txt', 'r')
    lines_in = file_in.read().split('\n')
    file_in.close()

    print('len lines_in : ', len(lines_in))

    lines_in = list(set(lines_in).difference(set(lines_train_clean)))
    lines_in = list(set(lines_in).difference(set(lines_train_unclean)))

    print('len lines_in : ', len(lines_in))

    file_pilot_1 = open('../benchmark_pilot_1/'+lang_code+'_test.txt', 'r')
    lines_pilot_1 = file_pilot_1.read().split('\n')
    file_pilot_1.close()

    lines_in = list(set(lines_in).difference(set(lines_pilot_1)))

    print('len lines_in : ', len(lines_in))


    lines_in = random.choices(lines_in, k = samples)
    
    file_out = open(lang_code+'_test.txt', 'w')
    file_out.write('\n'.join(lines_in))
    file_out.close()


lang_code_list = ['as', 'or']

for lang_code in lang_code_list:
    

    lines_train_clean = []

    if 'i' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_indiccorp/IndicCorp_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'n' in train_data_sources[lang_code]:
        train_file_name = '../../Other_sources/nllb_preprocess/nllb_data_subset_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'w' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_wikipedia/wikipedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'v' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_vikaspedia/vikaspedia_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'nws' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_news_crawl/news_tok_norm_romanized_100k_sample_cleaned/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()

    if 'a' in train_data_sources[lang_code]:
        train_file_name = '../../Other_sources/annotator_ai4bharat_prerocess/train_tok_norm_romanized_100k_sample_cleaned/'+lang_code+'/'+lang_code+'_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_clean += lines_train_in
        file_train_in.close()


    lines_train_unclean = []

    if 'i' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_indiccorp/IndicCorp_data_subset_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'n' in train_data_sources[lang_code]:
        train_file_name = '../../Other_sources/nllb_preprocess/nllb_data_subset_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'w' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_wikipedia/wikipedia_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'v' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_vikaspedia/vikaspedia_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'nws' in train_data_sources[lang_code]:
        train_file_name = '../../preprocess_news_crawl/news_tok_norm_romanized/' + lang_code + '/'+lang_code + '_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()

    if 'a' in train_data_sources[lang_code]:
        train_file_name = '../../Other_sources/annotator_ai4bharat_prerocess/train_tok_norm_romanized/'+lang_code+'/'+lang_code+'_indic_tok.txt'
        file_train_in = open(train_file_name, 'r')
        lines_train_in = file_train_in.read().split('\n')
        lines_train_unclean += lines_train_in
        file_train_in.close()



    file_in = open('../../preprocess_indiccorp/IndicCorp_data/'+lang_code+'/'+lang_code+'_combine.txt', 'r')
    lines_in = file_in.read().split('\n')
    file_in.close()

    print('len lines_in : ', len(lines_in))

    lines_in = list(set(lines_in).difference(set(lines_train_clean)))
    lines_in = list(set(lines_in).difference(set(lines_train_unclean)))

    print('len lines_in : ', len(lines_in))

    file_pilot_1 = open('../benchmark_pilot_1/'+lang_code+'_test.txt', 'r')
    lines_pilot_1 = file_pilot_1.read().split('\n')
    file_pilot_1.close()
    
    lines_in = list(set(lines_in).difference(set(lines_pilot_1)))

    print('len lines_in : ', len(lines_in))

    lines_in = random.choices(lines_in, k = samples)
    
    file_out = open(lang_code+'_test.txt', 'w')
    file_out.write('\n'.join(lines_in))
    file_out.close()