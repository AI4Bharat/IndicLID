import fasttext
fasttext_model_path = '/nlsasfs/home/ai4bharat/yashkm/yash/indic-lid/final_runs/roman_model/fasttext_word_embed/clean_samples/tune_run/basline_en_other/result_model_dim_64/embed.bin'
model = fasttext.load_model(fasttext_model_path)

word_samples = ['tumhne', 'tere']
word_neighbors_dict = {}
number_of_neighbors = 5
for word in word_samples:
    neighbors_tuple = model.get_nearest_neighbors(word)
    top_neighbors = []
    for neighbor in neighbors_tuple[:number_of_neighbors]:
        top_neighbors.append(neighbor[1])
    # print(top_neighbors)
    word_neighbors_dict[word] = top_neighbors

print(word_neighbors_dict)