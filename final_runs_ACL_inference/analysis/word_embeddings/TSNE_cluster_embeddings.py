import random
# import pandas as pd
# pd.options.mode.chained_assignment = None 
import numpy as np
# import re
# import nltk
import sklearn
# from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# %matplotlib inline

# train samples 
file_in_train = open('../../clean_corpus/train_combine.txt', 'r')
lines_in_train = file_in_train.read().split('\n')
# lines_in_train = [ ' '.join(line.split(' ')[1:]) for line in lines_in_train]

# lines_sampled = random.choices( lines_in_train, k=100 )

unique_word_lists_dict = {}
for line in lines_in_train:
    labels = line.split(' ')[0][9:]
    words = line.split(' ')[1:]
    if labels in unique_word_lists_dict:
        unique_word_lists_dict[labels] += words
    else:
        unique_word_lists_dict[labels] = words

words_sampled = []
for labels in unique_word_lists_dict:
    print(labels)
    print('total words : ', len(unique_word_lists_dict[labels]))
    print(type(set(unique_word_lists_dict[labels])))
    unique_word_lists_dict[labels] = list(set(unique_word_lists_dict[labels]))
    print('total unique words : ', len(unique_word_lists_dict[labels]))

    samples = random.choices( unique_word_lists_dict[labels], k=2 )
    words_sampled += [(sample, labels) for sample in samples]



# load fasttext model
import fasttext
fasttext_lid_model_path='/nlsasfs/home/ai4bharat/yashkm/yash/indic-lid/final_runs/roman_model/fasttext/clean_samples/tune_run/basline_en_other/result_model_dim_8/model_baseline_roman.bin'
fasttext_model = fasttext.load_model(fasttext_lid_model_path)

color_dict  = {
    'Assamese' : 'grey',
    'Bangla' : 'lightcoral',
    'Bodo' : 'brown',
    'Konkani' : 'red', 
    'Gujarati' : 'chocolate',
    'Hindi' : 'sienna',
    'Kannada' : 'orange',
    'Kashmiri' : 'goldenrod',
    'Maithili' : 'gold',
    'Malayalam' : 'olive',
    'Manipuri_Mei' : 'yellow',
    'Marathi' : 'greenyellow',
    'Nepali' : 'darkgreen',
    'Oriya' : 'green',
    'Punjabi' : 'teal',
    'Sanskrit' : 'aqua',
    'Sindhi' : 'deepskyblue',
    'Tamil' : 'dodgerblue',
    'Telugu' : 'blue',
    'Urdu' : 'indigo',
    'English' : 'violet',
    'Other' : 'pink'
}

def tsne_plot(fasttext_model, words_sampled):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    class_colours = []
    for word_tuple in words_sampled:
        word = word_tuple[0]
        lang = word_tuple[1]
        
        tokens.append(fasttext_model.get_word_vector(word))
        labels.append(word)
        class_colours.append(color_dict[lang])
    
    tokens = np.array(tokens)

    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=class_colours[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # plt.show()
    plt.savefig('../result/foo.png')

tsne_plot(fasttext_model, words_sampled)