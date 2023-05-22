import fasttext
import csv
import sys
# model_dim = int(sys.argv[1])

model = fasttext.load_model('../result/nllblid218e')



file = open('../result/result.txt', 'w')

lines_config = []

lines_config.append('lr : ' + str ( model.f.getArgs().lr ) )
lines_config.append('dim : ' + str ( model.f.getArgs().dim ) )
lines_config.append('ws : ' + str ( model.f.getArgs().ws ) )
lines_config.append('epoch : ' + str ( model.f.getArgs().epoch ) )
lines_config.append('minCount : ' + str ( model.f.getArgs().minCount ) )
lines_config.append('minCountLabel : ' + str ( model.f.getArgs().minCountLabel ) )
lines_config.append('minn : ' + str ( model.f.getArgs().minn ) )
lines_config.append('maxn : ' + str ( model.f.getArgs().maxn ) )
lines_config.append('neg : ' + str ( model.f.getArgs().neg ) )
lines_config.append('wordNgrams : ' + str ( model.f.getArgs().wordNgrams ) )
lines_config.append('loss : ' + str ( model.f.getArgs().loss ) )
lines_config.append('bucket : ' + str ( model.f.getArgs().bucket ) )
lines_config.append('thread : ' + str ( model.f.getArgs().thread ) )
lines_config.append('lrUpdateRate : ' + str ( model.f.getArgs().lrUpdateRate ) )
lines_config.append('t : ' + str ( model.f.getArgs().t ) )
lines_config.append('label : ' + str ( model.f.getArgs().label ) )
lines_config.append('verbose : ' + str ( model.f.getArgs().verbose ) )
lines_config.append('pretrainedVectors : ' + str ( model.f.getArgs().pretrainedVectors ) )




file.write('Hyperparameters (Tuned)')
file.write('\n')
file.write('\n'.join(lines_config))
file.write('\n\n\n\n\n')



# result_train = model.test("../corpus/train_combine.txt")

# file.write('Evaluation Results Train Set')
# file.write('\n')
# file.write('train Set')
# file.write('\n')
# file.write('Samples : ' + str(result_train[0]) )
# file.write('\n')
# file.write('precision : ' + str(result_train[1]) )
# file.write('\n')
# file.write('recall : ' + str(result_train[2]) )
# file.write('\n')
# file.write('F1-Score : ' + str( (2 * result_train[1] * result_train[2]) / (result_train[1] + result_train[2]) ) )
# file.write('\n\n\n')




# result_train = model.test("../corpus/valid_combine.txt")

# file.write('Evaluation Results valid_combine')
# file.write('\n')
# file.write('train Set')
# file.write('\n')
# file.write('Samples : ' + str(result_train[0]) )
# file.write('\n')
# file.write('precision : ' + str(result_train[1]) )
# file.write('\n')
# file.write('recall : ' + str(result_train[2]) )
# file.write('\n')
# file.write('F1-Score : ' + str( (2 * result_train[1] * result_train[2]) / (result_train[1] + result_train[2]) ) )
# file.write('\n\n\n')


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
    'English' : 24,
    'Other' : 25
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
    24 : 'English',
    25 : 'Other'
}


flores_reverse_dict = {
    'asm_Beng' : 'Assamese',
    'ben_Beng' : 'Bangla', 
    'guj_Gujr' : 'Gujarati',
    'hin_Deva' : 'Hindi', 
    'kan_Knda' : 'Kannada', 
    'mai_Deva' : 'Maithili', 
    'mal_Mlym' : 'Malayalam', 
    'mar_Deva' : 'Marathi', 
    'npi_Deva' : 'Nepali', 
    'ory_Orya' : 'Oriya', 
    'pan_Guru' : 'Punjabi', 
    'san_Deva' : 'Sanskrit', 
    'snd_Arab' : 'Sindhi', 
    'tam_Taml' : 'Tamil', 
    'tel_Telu' : 'Telugu', 
    'kas_Arab' : 'Kashmiri_Arab', 
    'kas_Deva' : 'Kashmiri_Deva',
    'mni_Beng' : 'Manipuri_Beng', 
    'urd_Arab' : 'Urdu',
    'eng_Latn' : 'English',
    'sat_Olck' : 'Santali'
}




def evaluate(test_file_name, languages):
    
    classes = 26

    # inference for Dakshina, test_combine_dakshina
    file_test = open('../corpus/'+test_file_name+'.txt', 'r')
    lines_test = file_test.read().split('\n')
    file_test.close()


    # save predictions
    file_predictions = open('../result/predictions_'+test_file_name+'.csv', 'w')
    csv_writer_predictions = csv.writer(file_predictions)
    csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )

    file_predictions_right = open('../result/right_predictions_'+test_file_name+'.csv', 'w')
    csv_writer_predictions_right = csv.writer(file_predictions_right)
    csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


    file_predictions_wrong = open('../result/wrong_predictions_'+test_file_name+'.csv', 'w')
    csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
    csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



    # Computing confusion matrix
    confusion_matrix = []
    for i in range(classes):
        confusion_matrix.append( [0]*classes )
        
    # to calculate accuracy and save right and wrong prediction
    count = 0
    n = 0
    for line in lines_test:
        label = line.split(' ')[0]
        sen = ' '.join(line.split(' ')[1:])
        pred_label = model.predict(sen)[0][0]
        pred_score = model.predict(sen)[1][0]
        

        if pred_label[9:] in flores_reverse_dict:
            pred_label = '__label__'+flores_reverse_dict[pred_label[9:]]
        else:
            pred_label = '__label__Other'


        if pred_label == label:
            count+=1
            csv_writer_predictions_right.writerow( [ sen, label, pred_label, pred_score ] )
        else:
            csv_writer_predictions_wrong.writerow( [ sen, label, pred_label, pred_score ] )
        n+=1   

        confusion_matrix[ confusion_matrix_mapping[label[9:]] ][ confusion_matrix_mapping[pred_label[9:]] ] += 1
        
        csv_writer_predictions.writerow( [ sen, label, pred_label, pred_score ] )




    # Computing precision, recall and f1
    precsison_recall_f1 = []
    for i in range(classes):
        precsison_recall_f1.append([0] * 3)

    # precision_denominator = 0
    # recall_denominator = 0
    # f1_denominator = 0

    for i in range(classes):
        no_of_correctly_predicted = confusion_matrix[i][i]
        total_predictions_as_i = 0

        precision = 0
        recall = 0
        f1_value = 0

        # true predicted i values out of all predicted i values
        for j in range(classes):
            total_predictions_as_i += confusion_matrix[j][i]
        if (total_predictions_as_i != 0):
            precision = no_of_correctly_predicted/total_predictions_as_i
            # precision_denominator += 1

        # true predicted i values out of all actual i values
        total_actual_values_of_i = sum(confusion_matrix[i])
        if (total_actual_values_of_i != 0):
            recall = no_of_correctly_predicted/total_actual_values_of_i
            # recall_denominator += 1
        
        # f1 score
        if (precision + recall != 0):
            f1_value = (2 * precision * recall) / (precision + recall)
            # f1_denominator += 1
            
        precsison_recall_f1[i][0] = precision
        precsison_recall_f1[i][1] = recall
        precsison_recall_f1[i][2] = f1_value

    avg_precision = sum([precsison_recall_f1[i][0] for i in range(classes)]) / languages
    avg_recall = sum([precsison_recall_f1[i][1] for i in range(classes)]) / languages
    avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(classes)]) / languages



    # to save confusion matrix and precision recall matrix
    
    for i in range(classes):
        precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

    precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
    precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )


    file_precision_recall_f1 = open('../result/precision_recall_f1_'+test_file_name+'.csv', 'w')
    precision_recall_f1_csv_writer = csv.writer(file_precision_recall_f1)

    for i in range(classes+2):
        precision_recall_f1_csv_writer.writerow(precsison_recall_f1[i])
    file_precision_recall_f1.close()



    # save confusion matrix
    for i in range(classes):
        confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 

    confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(classes)] )


    file_confusion_matrix = open('../result/confusion_matrix_'+test_file_name+'.csv', 'w')
    confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

    for i in range(classes+1):
        confusion_matrix_csv_writer.writerow(confusion_matrix[i])


    file_confusion_matrix.close()
    file_predictions.close()
    file_predictions_right.close()
    file_predictions_wrong.close()




    # fasttext evaluation scores 
    test_acc = count/n
    result_test = model.test('../corpus/'+test_file_name+'.txt')
    file.write('fasttext evaluation scores - ' + test_file_name + '\n')
    file.write('Samples : ' + str(result_test[0]) + '\n')
    file.write('precision : ' + str(result_test[1]) + '\n')
    file.write('recall : ' + str(result_test[2]) + '\n')
    file.write('F1-Score : ' + str( (2 * result_test[1] * result_test[2]) / (result_test[1] + result_test[2]) ) + '\n')
    file.write('Accuracy : ' + str(test_acc) + '\n')
    file.write('\n\n\n')


evaluate('test_combine_native_20', 20)


file.close()


