import fasttext
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
    'Urdu' : 19,
    'English' : 20,
    'Other' : 21
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
    19 : 'Urdu',
    20 : 'English',
    21 : 'Other'
}


def len_wise_analysis(file_name, step):
    
    file_predictions = open('../result/predictions_'+file_name+'.csv', 'r')
    lines_test_org = file_predictions.read().split('\n')[1:]


    file_predictions_right = open('../result/right_predictions_'+file_name+'_len_wise.csv', 'w')
    csv_writer_predictions_right = csv.writer(file_predictions_right)
    csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score', 'len_min', 'len_max' ] )


    file_predictions_wrong = open('../result/wrong_predictions_'+file_name+'_len_wise.csv', 'w')
    csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
    csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score', 'len_min', 'len_max' ] )


    lines_test = lines_test_org[:]

    lines_test_len = [len(line.split(' ')) for line in lines_test]
    
    max_len = max(lines_test_len)
    
    file_len = open('../result/'+file_name+'_len_wise_acc.txt', 'w')
    
    file_len.write('Total samples' + '\t' + 'len_min-len_max' + '\t' + 'Accuracy'+'\n')
    
    i=1
    while i < max_len:

        len_min = i
        len_max = len_min+step-1

        lines_test = lines_test_org[:]
                
        lines_test = [line for line in lines_test if len(line.split(' ')) <= len_max]
        
        lines_test = [line for line in lines_test if len(line.split(' ')) >= len_min]

        lines_test = [line for line in lines_test if line]
        
        # to calculate accuracy
        count = 0
        n = 0
        if lines_test:
            for line in lines_test:
                
                end_token = line.split(' ')[-1]
                
                pred_score = end_token.split(',')[-1]
                
                pred_label = end_token.split(',')[-2]
                
                label = end_token.split(',')[-3]
                
                sen_end_token =  end_token.split(',')[-4]

                sen = ' '.join( line.split(' ')[:-1] ) + ' ' + sen_end_token

                if pred_label == label:
                    count+=1
                    csv_writer_predictions_right.writerow( [ sen, label, pred_label, pred_score, len_min, len_max ] )
                else:
                    csv_writer_predictions_wrong.writerow( [ sen, label, pred_label, pred_score, len_min, len_max ] )
                n+=1

            file_len.write(str(n)+'\t'+str(len_min)+'-'+str(len_max)+'\t'+str(count/n)+'\n')

        i = i + step    
    
    file_len.close()
    file_predictions.close()
    file_predictions_right.close()
    file_predictions_wrong.close()

len_wise_analysis('test_combine_dakshina', 3)
len_wise_analysis('valid_combine_dakshina', 3)
len_wise_analysis('test_combine_flores200_romanized', 3)
len_wise_analysis('test_combine_ai4b_romanized', 3)

# len_wise_analysis('test_combine_dakshina_romanized', 3)

# def len_wise_analysis(file_name, len):

#     file = open('../result/result.txt', 'w')
#     # inference for dakshina_romanized
#     file_test = open('../corpus/test_combine_dakshina_romanized.txt', 'r')
#     lines_test = file_test.read().split('\n')

#     # save predictions
#     file_predictions = open('../result/predictions_dakshina_romanized.csv', 'w')
#     csv_writer_predictions = csv.writer(file_predictions)
#     csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



#     file_predictions_right = open('../result/right_predictions_dakshina_romanized.csv', 'w')
#     csv_writer_predictions_right = csv.writer(file_predictions_right)
#     csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


#     file_predictions_wrong = open('../result/wrong_predictions_dakshina_romanized.csv', 'w')
#     csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
#     csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



#     # save confusion matrix
#     confusion_matrix = []
#     for i in range(20):
#         confusion_matrix.append( [0]*20 )
        
#     # to calculate accuracy
#     count = 0
#     n = 0
#     for line in lines_test:
#         label = line.split(' ')[0]
#         sen = ' '.join(line.split(' ')[1:])
#         pred_label = model.predict(sen)[0][0]
#         pred_score = model.predict(sen)[1][0]
        
#         if pred_label == label:
#             count+=1
#             csv_writer_predictions_right.writerow( [ sen, label, pred_label, pred_score ] )
#         else:
#             csv_writer_predictions_wrong.writerow( [ sen, label, pred_label, pred_score ] )
#         n+=1   

#         confusion_matrix[ confusion_matrix_mapping[label[9:]] ][ confusion_matrix_mapping[pred_label[9:]] ] += 1
        
#         csv_writer_predictions.writerow( [ sen, label, pred_label, pred_score ] )








#     precsison_recall_f1 = []
#     for i in range(20):
#         precsison_recall_f1.append([0] * 3)

#     precision_denominator = 0
#     recall_denominator = 0
#     f1_denominator = 0

#     # Computing precision, recall and f1
#     for i in range(20):
#         no_of_correctly_predicted = confusion_matrix[i][i]
#         total_predictions_as_i = 0

#         precision = 0
#         recall = 0
#         f1_value = 0

#         # true predicted i values out of all predicted i values
#         for j in range(20):
#             total_predictions_as_i += confusion_matrix[j][i]
#         if (total_predictions_as_i != 0):
#             precision = no_of_correctly_predicted/total_predictions_as_i
#             precision_denominator += 1

#         # true predicted i values out of all actual i values
#         total_actual_values_of_i = sum(confusion_matrix[i])
#         if (total_actual_values_of_i != 0):
#             recall = no_of_correctly_predicted/total_actual_values_of_i
#             recall_denominator += 1
        
#         # f1 score
#         if (precision + recall != 0):
#             f1_value = (2 * precision * recall) / (precision + recall)
#             f1_denominator += 1
            
#         precsison_recall_f1[i][0] = precision
#         precsison_recall_f1[i][1] = recall
#         precsison_recall_f1[i][2] = f1_value

#     avg_precision = sum([precsison_recall_f1[i][0] for i in range(20)]) / precision_denominator
#     avg_recall = sum([precsison_recall_f1[i][1] for i in range(20)]) / recall_denominator
#     avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(20)]) / f1_denominator




#     for i in range(20):
#         confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
#         precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

#     confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(20)] )
#     precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
#     precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

#     file.write('Test Set (dakshina romanized) : Precision-Recall-F1-Matrix')
#     file.write('\n')
#     for i in range(22):
#         file.write( str(precsison_recall_f1[i][0]) + '\t' + str(precsison_recall_f1[i][1]) 
#                         + '\t' + str(precsison_recall_f1[i][2]) 
#                         + '\t' + str(precsison_recall_f1[i][3]) 
#                     ) 
#         file.write('\n')


#     test_acc = count/n

#     file_test.close()
#     file_predictions.close()
#     file_predictions_right.close()
#     file_predictions_wrong.close()

#     file_confusion_matrix = open('../result/confusion_matrix_dakshina_romanized.csv', 'w')
#     confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

#     for i in range(21):
#         confusion_matrix_csv_writer.writerow(confusion_matrix[i])

#     file_confusion_matrix.close()









#     result_test = model.test("../corpus/test_combine_dakshina_romanized.txt")


#     file.write('Test Set (dakshina romanized)')
#     file.write('\n')
#     file.write('Samples : ' + str(result_test[0]))
#     file.write('\n')
#     file.write('precision : ' + str(result_test[1]))
#     file.write('\n')
#     file.write('recall : ' + str(result_test[2]))
#     file.write('\n')
#     file.write('F1-Score : ' + str( (2 * result_test[1] * result_test[2]) / (result_test[1] + result_test[2]) ) )
#     file.write('\n')
#     file.write('Test accuracy : ' + str(test_acc))
#     file.write('\n\n\n')
#     file.close()
