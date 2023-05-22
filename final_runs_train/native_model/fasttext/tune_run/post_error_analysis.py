import statistics
import fasttext
import csv

def length():
    file_predictions_right = open('../result/right_predictions_dakshina_test.csv', 'r')
    lines_predictions_right = file_predictions_right.read().split('\n')[1:]
    lines_predictions_right = [line for line in lines_predictions_right if line]

    sen_right = [ ','.join(line.split(',')[:-3]) for line in lines_predictions_right]
    sen_right_len = [len(line.split(' ')) for line in sen_right]
    sen_right_score = [float(line.split(',')[-1].strip()) for line in lines_predictions_right]

    print("mean(sen_right_len) ",statistics.mean(sen_right_len))
    print("mode(sen_right_len) ",statistics.mode(sen_right_len))
    print("median(sen_right_len) ",statistics.median(sen_right_len))
    print("mean(sen_right_score) ",statistics.mean(sen_right_score))

    file_predictions_wrong = open('../result/wrong_predictions_dakshina_test.csv', 'r')
    lines_predictions_wrong = file_predictions_wrong.read().split('\n')[1:]
    lines_predictions_wrong = [line for line in lines_predictions_wrong if line]

    sen_wrong = [ ','.join(line.split(',')[:-3]) for line in lines_predictions_wrong]
    sen_wrong_len = [len(line.split(' ')) for line in sen_wrong]
    sen_wrong_score = [float(line.split(',')[-1].strip()) for line in lines_predictions_wrong]

    print("mean(sen_wrong_len) ",statistics.mean(sen_wrong_len))
    print("mode(sen_wrong_len) ",statistics.mode(sen_wrong_len))
    print("median(sen_wrong_len) ",statistics.median(sen_wrong_len))
    print("mean(sen_wrong_score) ",statistics.mean(sen_wrong_score))

    print(len(sen_wrong))
    lines_predictions_wrong_threshold_len  = [ line for line in sen_wrong if len(line.split(' ')) < 3 ]
    print(len(lines_predictions_wrong_threshold_len))

def train_data_errors():

    model = fasttext.load_model('../result/model_baseline_native.bin')

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

    # inference for Dakshina
    file_train = open('../corpus/train_combine.txt', 'r')
    lines_train = file_train.read().split('\n')

    # save predictions
    file_predictions = open('../result/predictions_train.csv', 'w')
    csv_writer_predictions = csv.writer(file_predictions)
    csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



    file_predictions_right = open('../result/right_predictions_train.csv', 'w')
    csv_writer_predictions_right = csv.writer(file_predictions_right)
    csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


    file_predictions_wrong = open('../result/wrong_predictions_train.csv', 'w')
    csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
    csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



    # save confusion matrix
    confusion_matrix = []
    for i in range(24):
        confusion_matrix.append( [0]*24 )
        
    # to calculate accuracy
    count = 0
    n = 0
    for line in lines_train:
        label = line.split(' ')[0]
        sen = ' '.join(line.split(' ')[1:])
        pred_label = model.predict(sen)[0][0]
        pred_score = model.predict(sen)[1][0]
        
        if pred_label == label:
            count+=1
            csv_writer_predictions_right.writerow( [ sen, label, pred_label, pred_score ] )
        else:
            csv_writer_predictions_wrong.writerow( [ sen, label, pred_label, pred_score ] )
        n+=1   

        confusion_matrix[ confusion_matrix_mapping[label[9:]] ][ confusion_matrix_mapping[pred_label[9:]] ] += 1
        
        csv_writer_predictions.writerow( [ sen, label, pred_label, pred_score ] )








    precsison_recall_f1 = []
    for i in range(24):
        precsison_recall_f1.append([0] * 3)

    precision_denominator = 0
    recall_denominator = 0
    f1_denominator = 0

    # Computing precision, recall and f1
    for i in range(24):
        no_of_correctly_predicted = confusion_matrix[i][i]
        total_predictions_as_i = 0

        precision = 0
        recall = 0
        f1_value = 0

        # true predicted i values out of all predicted i values
        for j in range(24):
            total_predictions_as_i += confusion_matrix[j][i]
        if (total_predictions_as_i != 0):
            precision = no_of_correctly_predicted/total_predictions_as_i
            precision_denominator += 1

        # true predicted i values out of all actual i values
        total_actual_values_of_i = sum(confusion_matrix[i])
        if (total_actual_values_of_i != 0):
            recall = no_of_correctly_predicted/total_actual_values_of_i
            recall_denominator += 1
        
        # f1 score
        if (precision + recall != 0):
            f1_value = (2 * precision * recall) / (precision + recall)
            f1_denominator += 1
            
        precsison_recall_f1[i][0] = precision
        precsison_recall_f1[i][1] = recall
        precsison_recall_f1[i][2] = f1_value

    avg_precision = sum([precsison_recall_f1[i][0] for i in range(24)]) / precision_denominator
    avg_recall = sum([precsison_recall_f1[i][1] for i in range(24)]) / recall_denominator
    avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(24)]) / f1_denominator




    for i in range(24):
        confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
        precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

    confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(24)] )
    precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
    precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

    # file.write('Train set : Precision-Recall-F1-Matrix')
    # file.write('\n')
    # for i in range(26):
    #     file.write( str(precsison_recall_f1[i][0]) + '\t' + str(precsison_recall_f1[i][1]) 
    #                     + '\t' + str(precsison_recall_f1[i][2]) 
    #                     + '\t' + str(precsison_recall_f1[i][3]) 
    #                 ) 
    #     file.write('\n')
    print(precsison_recall_f1)

    test_acc = count/n

    # file_test.close()
    file_predictions.close()
    file_predictions_right.close()
    file_predictions_wrong.close()

    file_confusion_matrix = open('../result/confusion_matrix_train.csv', 'w')
    confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

    for i in range(25):
        confusion_matrix_csv_writer.writerow(confusion_matrix[i])

    file_confusion_matrix.close()

train_data_errors()