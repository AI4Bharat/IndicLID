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
    autotuneDuration=14400
    )
model.save_model("../result_model_dim_" + str(model_dim) + "/model_baseline_roman.bin")
# model = fasttext.load_model('../result/model_baseline_12.bin')


# file = open('../result/result.txt', 'w')

# lines_config = []

# lines_config.append('lr : ' + str ( model.f.getArgs().lr ) )
# lines_config.append('dim : ' + str ( model.f.getArgs().dim ) )
# lines_config.append('ws : ' + str ( model.f.getArgs().ws ) )
# lines_config.append('epoch : ' + str ( model.f.getArgs().epoch ) )
# lines_config.append('minCount : ' + str ( model.f.getArgs().minCount ) )
# lines_config.append('minCountLabel : ' + str ( model.f.getArgs().minCountLabel ) )
# lines_config.append('minn : ' + str ( model.f.getArgs().minn ) )
# lines_config.append('maxn : ' + str ( model.f.getArgs().maxn ) )
# lines_config.append('neg : ' + str ( model.f.getArgs().neg ) )
# lines_config.append('wordNgrams : ' + str ( model.f.getArgs().wordNgrams ) )
# lines_config.append('loss : ' + str ( model.f.getArgs().loss ) )
# lines_config.append('bucket : ' + str ( model.f.getArgs().bucket ) )
# lines_config.append('thread : ' + str ( model.f.getArgs().thread ) )
# lines_config.append('lrUpdateRate : ' + str ( model.f.getArgs().lrUpdateRate ) )
# lines_config.append('t : ' + str ( model.f.getArgs().t ) )
# lines_config.append('label : ' + str ( model.f.getArgs().label ) )
# lines_config.append('verbose : ' + str ( model.f.getArgs().verbose ) )
# lines_config.append('pretrainedVectors : ' + str ( model.f.getArgs().pretrainedVectors ) )




# file.write('Hyperparameters (Tuned)')
# file.write('\n')
# file.write('\n'.join(lines_config))
# file.write('\n\n\n\n\n')



# result_train = model.test("../corpus/train_combine.txt")

# file.write('Evaluation Results')
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





# confusion_matrix_mapping  = {
#     'Assamese' : 0,
#     'Bangla' : 1,
#     'Bodo' : 2,
#     'Dogri' : 3,
#     'Konkani' : 4, 
#     'Gujarati' : 5,
#     'Hindi' : 6,
#     'Kannada' : 7,
#     'Kashmiri_Arab' : 8,
#     'Kashmiri_Deva' : 9,
#     'Maithili' : 10,
#     'Malayalam' : 11,
#     'Manipuri_Beng' : 12,
#     'Manipuri_Mei' : 13,
#     'Marathi' : 14,
#     'Nepali' : 15,
#     'Oriya' : 16,
#     'Punjabi' : 17,
#     'Sanskrit' : 18,
#     'Santali' : 19,
#     'Sindhi' : 20,
#     'Tamil' : 21,
#     'Telugu' : 22,
#     'Urdu' : 23,
# }

# confusion_matrix_reverse_mapping  = {
#     0 : 'Assamese',
#     1 : 'Bangla',
#     2 : 'Bodo',
#     3 : 'Dogri',
#     4 : 'Konkani', 
#     5 : 'Gujarati',
#     6 : 'Hindi',
#     7 : 'Kannada',
#     8 : 'Kashmiri_Arab',
#     9 : 'Kashmiri_Deva',
#     10 : 'Maithili',
#     11 : 'Malayalam',
#     12 : 'Manipuri_Beng',
#     13 : 'Manipuri_Mei',
#     14 : 'Marathi',
#     15 : 'Nepali',
#     16 : 'Oriya',
#     17 : 'Punjabi',
#     18 : 'Sanskrit',
#     19 : 'Santali',
#     20 : 'Sindhi',
#     21 : 'Tamil',
#     22 : 'Telugu',
#     23 : 'Urdu',
# }

# # inference for Dakshina
# file_test = open('../corpus/test_combine_dakshina.txt', 'r')
# lines_test = file_test.read().split('\n')

# # save predictions
# file_predictions = open('../result/predictions_dakshina_test.csv', 'w')
# csv_writer_predictions = csv.writer(file_predictions)
# csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# file_predictions_right = open('../result/right_predictions_dakshina_test.csv', 'w')
# csv_writer_predictions_right = csv.writer(file_predictions_right)
# csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_wrong = open('../result/wrong_predictions_dakshina_test.csv', 'w')
# csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
# csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# # save confusion matrix
# confusion_matrix = []
# for i in range(24):
#     confusion_matrix.append( [0]*24 )
    
# # to calculate accuracy
# count = 0
# n = 0
# for line in lines_test:
#     label = line.split(' ')[0]
#     sen = ' '.join(line.split(' ')[1:])
#     pred_label = model.predict(sen)[0][0]
#     pred_score = model.predict(sen)[1][0]
    
#     if pred_label == label:
#         count+=1
#         csv_writer_predictions_right.writerow( [ sen, label, pred_label, pred_score ] )
#     else:
#         csv_writer_predictions_wrong.writerow( [ sen, label, pred_label, pred_score ] )
#     n+=1   

#     confusion_matrix[ confusion_matrix_mapping[label[9:]] ][ confusion_matrix_mapping[pred_label[9:]] ] += 1
    
#     csv_writer_predictions.writerow( [ sen, label, pred_label, pred_score ] )








# precsison_recall_f1 = []
# for i in range(24):
#     precsison_recall_f1.append([0] * 3)

# precision_denominator = 0
# recall_denominator = 0
# f1_denominator = 0

# # Computing precision, recall and f1
# for i in range(24):
#     no_of_correctly_predicted = confusion_matrix[i][i]
#     total_predictions_as_i = 0

#     precision = 0
#     recall = 0
#     f1_value = 0

#     # true predicted i values out of all predicted i values
#     for j in range(24):
#         total_predictions_as_i += confusion_matrix[j][i]
#     if (total_predictions_as_i != 0):
#         precision = no_of_correctly_predicted/total_predictions_as_i
#         precision_denominator += 1

#     # true predicted i values out of all actual i values
#     total_actual_values_of_i = sum(confusion_matrix[i])
#     if (total_actual_values_of_i != 0):
#         recall = no_of_correctly_predicted/total_actual_values_of_i
#         recall_denominator += 1
    
#     # f1 score
#     if (precision + recall != 0):
#         f1_value = (2 * precision * recall) / (precision + recall)
#         f1_denominator += 1
        
#     precsison_recall_f1[i][0] = precision
#     precsison_recall_f1[i][1] = recall
#     precsison_recall_f1[i][2] = f1_value

# avg_precision = sum([precsison_recall_f1[i][0] for i in range(24)]) / precision_denominator
# avg_recall = sum([precsison_recall_f1[i][1] for i in range(24)]) / recall_denominator
# avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(24)]) / f1_denominator




# for i in range(24):
#     confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
#     precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

# confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(24)] )
# precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
# precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

# file.write('Test Set (Dakshina) : Precision-Recall-F1-Matrix')
# file.write('\n')
# for i in range(26):
#     file.write( str(precsison_recall_f1[i][0]) + '\t' + str(precsison_recall_f1[i][1]) 
#                     + '\t' + str(precsison_recall_f1[i][2]) 
#                     + '\t' + str(precsison_recall_f1[i][3]) 
#                 ) 
#     file.write('\n')


# test_acc = count/n

# file_test.close()
# file_predictions.close()
# file_predictions_right.close()
# file_predictions_wrong.close()

# file_confusion_matrix = open('../result/confusion_matrix_dakshina_test.csv', 'w')
# confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

# for i in range(25):
#     confusion_matrix_csv_writer.writerow(confusion_matrix[i])

# file_confusion_matrix.close()











# result_valid = model.test("../corpus/valid_combine_dakshina.txt")


# file.write('Validation Set (Dakshina)')
# file.write('\n')
# file.write('Samples : ' + str(result_valid[0]))
# file.write('\n')
# file.write('precision : ' + str(result_valid[1]))
# file.write('\n')
# file.write('recall : ' + str(result_valid[2]))
# file.write('\n')
# file.write('F1-Score : ' + str( (2 * result_valid[1] * result_valid[2]) / (result_valid[1] + result_valid[2]) ) )
# file.write('\n\n\n')

# result_test = model.test("../corpus/test_combine_dakshina.txt")


# file.write('Test Set (Dakshina)')
# file.write('\n')
# file.write('Samples : ' + str(result_test[0]))
# file.write('\n')
# file.write('precision : ' + str(result_test[1]))
# file.write('\n')
# file.write('recall : ' + str(result_test[2]))
# file.write('\n')
# file.write('F1-Score : ' + str( (2 * result_test[1] * result_test[2]) / (result_test[1] + result_test[2]) ) )
# file.write('\n')
# file.write('Test accuracy : ' + str(test_acc))
# file.write('\n\n\n')











# # inference for Flores200

# file_test = open('../corpus/test_combine_flores200.txt', 'r')
# lines_test = file_test.read().split('\n')

# file_predictions = open('../result/predictions_flores200_test.csv', 'w')
# csv_writer_predictions = csv.writer(file_predictions)
# csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_right = open('../result/right_predictions_flores200_test.csv', 'w')
# csv_writer_predictions_right = csv.writer(file_predictions_right)
# csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_wrong = open('../result/wrong_predictions_flores200_test.csv', 'w')
# csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
# csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# confusion_matrix = []
# for i in range(24):
#     confusion_matrix.append( [0]*24 )

# # to calculate accuracy
# count = 0
# n = 0
# for line in lines_test:
#     label = line.split(' ')[0]
#     sen = ' '.join(line.split(' ')[1:])
#     pred_label = model.predict(sen)[0][0]
#     pred_score = model.predict(sen)[1][0]
    
#     if pred_label == label:
#         count+=1
#         csv_writer_predictions_right.writerow( [ sen, label, pred_label, pred_score ] )
#     else:
#         csv_writer_predictions_wrong.writerow( [ sen, label, pred_label, pred_score ] )
#     n+=1    

#     confusion_matrix[ confusion_matrix_mapping[label[9:]] ][ confusion_matrix_mapping[pred_label[9:]] ] += 1
    
#     csv_writer_predictions.writerow( [ sen, label, pred_label, pred_score ] )








# precsison_recall_f1 = []
# for i in range(24):
#     precsison_recall_f1.append([0] * 3)

# precision_denominator = 0
# recall_denominator = 0
# f1_denominator = 0

# # Computing precision, recall and f1
# for i in range(24):
#     no_of_correctly_predicted = confusion_matrix[i][i]
#     total_predictions_as_i = 0

#     precision = 0
#     recall = 0
#     f1_value = 0

#     # true predicted i values out of all predicted i values
#     for j in range(24):
#         total_predictions_as_i += confusion_matrix[j][i]
#     if (total_predictions_as_i != 0):
#         precision = no_of_correctly_predicted/total_predictions_as_i
#         precision_denominator += 1

#     # true predicted i values out of all actual i values
#     total_actual_values_of_i = sum(confusion_matrix[i])
#     if (total_actual_values_of_i != 0):
#         recall = no_of_correctly_predicted/total_actual_values_of_i
#         recall_denominator += 1
    
#     # f1 score
#     if (precision + recall != 0):
#         f1_value = (2 * precision * recall) / (precision + recall)
#         f1_denominator += 1
        
#     precsison_recall_f1[i][0] = precision
#     precsison_recall_f1[i][1] = recall
#     precsison_recall_f1[i][2] = f1_value

# avg_precision = sum([precsison_recall_f1[i][0] for i in range(24)]) / precision_denominator
# avg_recall = sum([precsison_recall_f1[i][1] for i in range(24)]) / recall_denominator
# avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(24)]) / f1_denominator




# for i in range(24):
#     confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
#     precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

# confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(24)] )
# precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
# precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

# file.write('Test Set (Flores200) : Precision-Recall-F1-Matrix')
# file.write('\n')
# for i in range(26):
#     file.write( str(precsison_recall_f1[i][0]) + '\t' + str(precsison_recall_f1[i][1]) 
#                     + '\t' + str(precsison_recall_f1[i][2]) 
#                     + '\t' + str(precsison_recall_f1[i][3]) 
#                 ) 
#     file.write('\n')









# test_acc = count/n

# file_test.close()
# file_predictions.close()
# file_predictions_right.close()
# file_predictions_wrong.close()


# file_confusion_matrix = open('../result/confusion_matrix_flores200_test.csv', 'w')
# confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

# for i in range(25):
#     confusion_matrix_csv_writer.writerow(confusion_matrix[i])

# file_confusion_matrix.close()











# result_valid = model.test("../corpus/valid_combine_flores200.txt")


# file.write('Validation Set (Flores200)')
# file.write('\n')
# file.write('Samples : ' + str(result_valid[0]))
# file.write('\n')
# file.write('precision : ' + str(result_valid[1]))
# file.write('\n')
# file.write('recall : ' + str(result_valid[2]))
# file.write('\n')
# file.write('F1-Score : ' + str( (2 * result_valid[1] * result_valid[2]) / (result_valid[1] + result_valid[2]) ) )
# file.write('\n\n\n')

# result_test = model.test("../corpus/test_combine_flores200.txt")


# file.write('Test Set (Flores200)')
# file.write('\n')
# file.write('Samples : ' + str(result_test[0]))
# file.write('\n')
# file.write('precision : ' + str(result_test[1]))
# file.write('\n')
# file.write('recall : ' + str(result_test[2]))
# file.write('\n')
# file.write('F1-Score : ' + str( (2 * result_test[1] * result_test[2]) / (result_test[1] + result_test[2]) ) )
# file.write('\n')
# file.write('Test accuracy : ' + str(test_acc))
# file.write('\n\n\n')












# # inference for AI4Bharat

# file_test = open('../corpus/test_combine_AI4Bharat.txt', 'r')
# lines_test = file_test.read().split('\n')

# file_predictions = open('../result/predictions_AI4Bharat_test.csv', 'w')
# csv_writer_predictions = csv.writer(file_predictions)
# csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_right = open('../result/right_predictions_AI4Bharat_test.csv', 'w')
# csv_writer_predictions_right = csv.writer(file_predictions_right)
# csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_wrong = open('../result/wrong_predictions_AI4Bharat_test.csv', 'w')
# csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
# csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# confusion_matrix = []
# for i in range(24):
#     confusion_matrix.append( [0]*24 )

# # to calculate accuracy
# count = 0
# n = 0
# for line in lines_test:
#     label = line.split(' ')[0]
#     sen = ' '.join(line.split(' ')[1:])
#     pred_label = model.predict(sen)[0][0]
#     pred_score = model.predict(sen)[1][0]
    
#     if pred_label == label:
#         count+=1
#         csv_writer_predictions_right.writerow( [ sen, label, pred_label, pred_score ] )
#     else:
#         csv_writer_predictions_wrong.writerow( [ sen, label, pred_label, pred_score ] )
#     n+=1   

#     confusion_matrix[ confusion_matrix_mapping[label[9:]] ][ confusion_matrix_mapping[pred_label[9:]] ] += 1
    
#     csv_writer_predictions.writerow( [ sen, label, pred_label, pred_score ] )







# precsison_recall_f1 = []
# for i in range(24):
#     precsison_recall_f1.append([0] * 3)

# precision_denominator = 0
# recall_denominator = 0
# f1_denominator = 0

# # Computing precision, recall and f1
# for i in range(24):
#     no_of_correctly_predicted = confusion_matrix[i][i]
#     total_predictions_as_i = 0

#     precision = 0
#     recall = 0
#     f1_value = 0

#     # true predicted i values out of all predicted i values
#     for j in range(24):
#         total_predictions_as_i += confusion_matrix[j][i]
#     if (total_predictions_as_i != 0):
#         precision = no_of_correctly_predicted/total_predictions_as_i
#         precision_denominator += 1

#     # true predicted i values out of all actual i values
#     total_actual_values_of_i = sum(confusion_matrix[i])
#     if (total_actual_values_of_i != 0):
#         recall = no_of_correctly_predicted/total_actual_values_of_i
#         recall_denominator += 1
    
#     # f1 score
#     if (precision + recall != 0):
#         f1_value = (2 * precision * recall) / (precision + recall)
#         f1_denominator += 1
        
#     precsison_recall_f1[i][0] = precision
#     precsison_recall_f1[i][1] = recall
#     precsison_recall_f1[i][2] = f1_value

# avg_precision = sum([precsison_recall_f1[i][0] for i in range(24)]) / precision_denominator
# avg_recall = sum([precsison_recall_f1[i][1] for i in range(24)]) / recall_denominator
# avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(24)]) / f1_denominator




# for i in range(24):
#     confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
#     precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

# confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(24)] )
# precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
# precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

# file.write('Test Set (AI4Bharat) : Precision-Recall-F1-Matrix')
# file.write('\n')
# for i in range(26):
#     file.write( str(precsison_recall_f1[i][0]) + '\t' + str(precsison_recall_f1[i][1]) 
#                     + '\t' + str(precsison_recall_f1[i][2]) 
#                     + '\t' + str(precsison_recall_f1[i][3]) 
#                 ) 
#     file.write('\n')




# test_acc = count/n

# file_test.close()
# file_predictions.close()
# file_predictions_right.close()
# file_predictions_wrong.close()


# file_confusion_matrix = open('../result/confusion_matrix_Ai4bharat_test.csv', 'w')
# confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

# for i in range(25):
#     confusion_matrix_csv_writer.writerow(confusion_matrix[i])

# file_confusion_matrix.close()










# result_valid = model.test("../corpus/valid_combine_AI4Bharat.txt")


# file.write('Validation Set (AI4Bharat)')
# file.write('\n')
# file.write('Samples : ' + str(result_valid[0]))
# file.write('\n')
# file.write('precision : ' + str(result_valid[1]))
# file.write('\n')
# file.write('recall : ' + str(result_valid[2]))
# file.write('\n')
# file.write('F1-Score : ' + str( (2 * result_valid[1] * result_valid[2]) / (result_valid[1] + result_valid[2]) ) )
# file.write('\n\n\n')

# result_test = model.test("../corpus/test_combine_AI4Bharat.txt")


# file.write('Test Set (AI4Bharat)')
# file.write('\n')
# file.write('Samples : ' + str(result_test[0]))
# file.write('\n')
# file.write('precision : ' + str(result_test[1]))
# file.write('\n')
# file.write('recall : ' + str(result_test[2]))
# file.write('\n')
# file.write('F1-Score : ' + str( (2 * result_test[1] * result_test[2]) / (result_test[1] + result_test[2]) ) )
# file.write('\n')
# file.write('Test accuracy : ' + str(test_acc))
# file.write('\n\n\n')
# file.close()
