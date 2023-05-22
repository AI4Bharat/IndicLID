import fasttext
import csv
from tqdm import tqdm

dimension = 256
embed_model = fasttext.train_unsupervised(
        input = '../corpus/train_combine.txt',
        minn=3, 
        maxn=6, 
        dim=dimension,
        model='skipgram',
        lr=0.09
        )
embed_model.save_model("../result/embed.bin")

def bin_to_vec(embed_model):
    
    # original BIN model loading
    f = embed_model

    # get all words from model
    words = f.get_words()

    with open('../result/embed.vec','w') as file_out:
        
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in words:
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr+'\n')
            except:
                pass


# embed_model_path = "../result/embed.bin"
# bin_to_vec(fasttext.load_model(embed_model_path))
# bin_to_vec(embed_model)



confusion_matrix_mapping  = {
    'Assamese' : 0,
    'Bangla' : 1,
    'Bodo' : 2,
    'Konkani' : 3, 
    'Gujarati' : 4,
    'Hindi' : 5,
    'Kannada' : 6,
    'Kashmiri' : 7,
    'Maithili' : 8,
    'Malayalam' : 9,
    'Manipuri_Mei' : 10,
    'Marathi' : 11,
    'Nepali' : 12,
    'Oriya' : 13,
    'Punjabi' : 14,
    'Sanskrit' : 15,
    'Sindhi' : 16,
    'Tamil' : 17,
    'Telugu' : 18,
    'Urdu' : 19
}

confusion_matrix_reverse_mapping  = {
    0 : 'Assamese',
    1 : 'Bangla',
    2 : 'Bodo',
    3 : 'Konkani', 
    4 : 'Gujarati',
    5 : 'Hindi',
    6 : 'Kannada',
    7 : 'Kashmiri',
    8 : 'Maithili',
    9 : 'Malayalam',
    10 : 'Manipuri_Mei',
    11 : 'Marathi',
    12 : 'Nepali',
    13 : 'Oriya',
    14 : 'Punjabi',
    15 : 'Sanskrit',
    16 : 'Sindhi',
    17 : 'Tamil',
    18 : 'Telugu',
    19 :  'Urdu'
}





# save embeddings
import torch


lines_train = open('../corpus/train_combine.txt', 'r').read().split('\n')
lines_test = open('../corpus/test_combine.txt', 'r').read().split('\n')
lines_valid = open('../corpus/valid_combine.txt', 'r').read().split('\n')

lines_valid_train_distribution = open('../corpus/valid_train_set_distribution.txt', 'r').read().split('\n')
lines_test_dakshina = open('../corpus/test_combine_dakshina.txt', 'r').read().split('\n')
lines_valid_dakshina = open('../corpus/valid_combine_dakshina.txt', 'r').read().split('\n')
lines_flores_romanized = open('../corpus/test_combine_flores200_romanized.txt', 'r').read().split('\n')
lines_dakshina_filtered = open('../../../../Dakshina/dakshina_filtered/test_combine_dakshina.txt', 'r').read().split('\n')

def create_sen_embedding(lines, model):
    labels = []
    sen_embed = torch.tensor([ 0 for _ in range(dimension) ], dtype = torch.float)


    line = lines[0]
    labels.append( confusion_matrix_mapping[ line.split(' ')[0][9:] ] )

    for word in line.split(' ')[1:]:
        sen_embed += torch.tensor(model.get_word_vector(word), dtype = float)
        
    sentence_embedding = sen_embed.view(1, -1)

    pbar = tqdm(total = len(lines))

    for line in lines[1:]:
        
        sen_embed = torch.tensor([ 0 for _ in range(dimension) ], dtype = torch.float)

        labels.append( confusion_matrix_mapping[ line.split(' ')[0][9:] ] )

        for word in line.split(' ')[1:]:
            sen_embed += torch.tensor(model.get_word_vector(word), dtype = float)
        
        sentence_embedding = torch.vstack((sentence_embedding, sen_embed))
        pbar.update(1)

    labels = torch.tensor(labels, dtype = int).view(-1, 1)
    pbar.close()
    
    return sentence_embedding, labels






train_X, train_y = create_sen_embedding(lines_train, embed_model)
train_tensor = torch.hstack((train_X, train_y))
torch.save(train_tensor, '../corpus/tensor_train.pt')

test_X, test_y = create_sen_embedding(lines_test, embed_model)
test_tensor = torch.hstack((test_X, test_y))
torch.save(test_tensor, '../corpus/tensor_test.pt')

valid_X, valid_y = create_sen_embedding(lines_valid, embed_model)
valid_tensor = torch.hstack((valid_X, valid_y))
torch.save(valid_tensor, '../corpus/tensor_valid.pt')



def save_embedding(lines, embed_model, file_name):
    text, labels = create_sen_embedding(lines, embed_model)
    data_tensor = torch.hstack((text, labels))
    torch.save(data_tensor, '../corpus/'+file_name+'.pt')

save_embedding(lines_valid_train_distribution, embed_model, 'valid_train_distribution')
save_embedding(lines_test_dakshina, embed_model, 'test_dakshina')
save_embedding(lines_valid_dakshina, embed_model, 'valid_dakshina')
save_embedding(lines_flores_romanized, embed_model, 'flores_romanized')
save_embedding(lines_dakshina_filtered, embed_model, 'dakshina_filtered')