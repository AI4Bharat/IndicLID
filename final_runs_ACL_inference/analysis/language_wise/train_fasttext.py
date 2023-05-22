import fasttext
import csv
import sys
model_dim = int(sys.argv[1])
model = fasttext.train_supervised(
    input = '../corpus/train_combine.txt', 
    loss = 'hs',
    verbose=1,
    dim = model_dim,
    autotuneValidationFile='../corpus/valid_combine.txt', 
    autotuneDuration=43200
    )
model.save_model("../result/model_baseline_roman.bin")
