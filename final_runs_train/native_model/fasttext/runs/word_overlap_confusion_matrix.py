import csv

confusion_matrix_mapping  = {
    'Assamese' : 0,
    'Bangla' : 1,
    'Bodo' : 2,
    'Dogri' : 3,
    'Konkani' : 4, 
    'Gujarati' : 5,
    'Hindi' : 6,
    'Kannada' : 7,
    'Kashmiri_Arab' : 8,
    'Kashmiri_Deva' : 9,
    'Maithili' : 10,
    'Malayalam' : 11,
    'Manipuri_Beng' : 12,
    'Manipuri_Mei' : 13,
    'Marathi' : 14,
    'Nepali' : 15,
    'Oriya' : 16,
    'Punjabi' : 17,
    'Sanskrit' : 18,
    'Santali' : 19,
    'Sindhi' : 20,
    'Tamil' : 21,
    'Telugu' : 22,
    'Urdu' : 23,
}

confusion_matrix_reverse_mapping  = {
    0 : 'Assamese',
    1 : 'Bangla',
    2 : 'Bodo',
    3 : 'Dogri',
    4 : 'Konkani', 
    5 : 'Gujarati',
    6 : 'Hindi',
    7 : 'Kannada',
    8 : 'Kashmiri_Arab',
    9 : 'Kashmiri_Deva',
    10 : 'Maithili',
    11 : 'Malayalam',
    12 : 'Manipuri_Beng',
    13 : 'Manipuri_Mei',
    14 : 'Marathi',
    15 : 'Nepali',
    16 : 'Oriya',
    17 : 'Punjabi',
    18 : 'Sanskrit',
    19 : 'Santali',
    20 : 'Sindhi',
    21 : 'Tamil',
    22 : 'Telugu',
    23 : 'Urdu',
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


    classes = 24

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
