import csv
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
    19 : 'Urdu'
}

def create_word_overlap_confusion_matrix(file_name):
    file = open('../corpus/'+file_name+'.txt', 'r')
    lines = file.read().split('\n')
    file.close()

    dict_words_langs = {}

    for line in lines:

        label = line.split(' ')[0][9:]

        if label in confusion_matrix_mapping.keys():
            for word in line.split(' ')[1:]:
                if word in dict_words_langs:
                    if label not in dict_words_langs[word]:
                        dict_words_langs[word].append(label)
                else:
                    dict_words_langs[word] = [label]


    classes = 20

    # save confusion matrix
    confusion_matrix = []
    for i in range(classes):
        confusion_matrix.append( [0]*classes )
        

    for word in dict_words_langs:
        for curr_label in dict_words_langs[word]:
            for loop_label in dict_words_langs[word]:
                confusion_matrix[confusion_matrix_mapping[curr_label]][confusion_matrix_mapping[loop_label]] += 1


    for i in range(classes):
        confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 

    confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(classes)] )



    file_confusion_matrix = open('../result/'+file_name+'_confusion_matrix_word_overlap.csv', 'w')
    confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

    for i in range(classes+1):
        confusion_matrix_csv_writer.writerow(confusion_matrix[i])

    file_confusion_matrix.close()


create_word_overlap_confusion_matrix('train_combine')
