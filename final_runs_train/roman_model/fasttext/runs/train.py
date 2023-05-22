import fasttext
import csv

model = fasttext.train_supervised(
    input = '../corpus/train_combine.txt', 
    lr = 0.45,
    dim = 512,
    ws = 5,
    epoch = 1,
    minCount = 1,
    minCountLabel = 0,
    minn = 3,
    maxn = 6,
    neg = 5,
    wordNgrams = 1,
    loss = 'hs',
    lrUpdateRate = 100,
    t = 0.0001,
    verbose = 1
    )
model.save_model("../result/model_baseline_roman.bin")
# model = fasttext.load_model('../result/model_baseline_12.bin')







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












# # inference for Flores200_romanized
# file_test = open('../corpus/test_combine_flores200_romanized.txt', 'r')
# lines_test = file_test.read().split('\n')

# # save predictions
# file_predictions = open('../result/predictions_flores200_romanized.csv', 'w')
# csv_writer_predictions = csv.writer(file_predictions)
# csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# file_predictions_right = open('../result/right_predictions_flores200_romanized.csv', 'w')
# csv_writer_predictions_right = csv.writer(file_predictions_right)
# csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_wrong = open('../result/wrong_predictions_flores200_romanized.csv', 'w')
# csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
# csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# # save confusion matrix
# confusion_matrix = []
# for i in range(20):
#     confusion_matrix.append( [0]*20 )
    
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
# for i in range(20):
#     precsison_recall_f1.append([0] * 3)

# precision_denominator = 0
# recall_denominator = 0
# f1_denominator = 0

# # Computing precision, recall and f1
# for i in range(20):
#     no_of_correctly_predicted = confusion_matrix[i][i]
#     total_predictions_as_i = 0

#     precision = 0
#     recall = 0
#     f1_value = 0

#     # true predicted i values out of all predicted i values
#     for j in range(20):
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

# avg_precision = sum([precsison_recall_f1[i][0] for i in range(20)]) / precision_denominator
# avg_recall = sum([precsison_recall_f1[i][1] for i in range(20)]) / recall_denominator
# avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(20)]) / f1_denominator




# for i in range(20):
#     confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
#     precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

# confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(20)] )
# precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
# precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

# file.write('Test Set (Flores_romanized) : Precision-Recall-F1-Matrix')
# file.write('\n')
# for i in range(22):
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

# file_confusion_matrix = open('../result/confusion_matrix_flores200_romanized.csv', 'w')
# confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

# for i in range(21):
#     confusion_matrix_csv_writer.writerow(confusion_matrix[i])

# file_confusion_matrix.close()









# result_test = model.test("../corpus/test_combine_flores200_romanized.txt")


# file.write('Test Set (Flores200_romanized)')
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














# # inference for ai4b_romanized
# file_test = open('../corpus/test_combine_ai4b_romanized.txt', 'r')
# lines_test = file_test.read().split('\n')

# # save predictions
# file_predictions = open('../result/predictions_ai4b_romanized.csv', 'w')
# csv_writer_predictions = csv.writer(file_predictions)
# csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# file_predictions_right = open('../result/right_predictions_ai4b_romanized.csv', 'w')
# csv_writer_predictions_right = csv.writer(file_predictions_right)
# csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_wrong = open('../result/wrong_predictions_ai4b_romanized.csv', 'w')
# csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
# csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# # save confusion matrix
# confusion_matrix = []
# for i in range(20):
#     confusion_matrix.append( [0]*20 )
    
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
# for i in range(20):
#     precsison_recall_f1.append([0] * 3)

# precision_denominator = 0
# recall_denominator = 0
# f1_denominator = 0

# # Computing precision, recall and f1
# for i in range(20):
#     no_of_correctly_predicted = confusion_matrix[i][i]
#     total_predictions_as_i = 0

#     precision = 0
#     recall = 0
#     f1_value = 0

#     # true predicted i values out of all predicted i values
#     for j in range(20):
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

# avg_precision = sum([precsison_recall_f1[i][0] for i in range(20)]) / precision_denominator
# avg_recall = sum([precsison_recall_f1[i][1] for i in range(20)]) / recall_denominator
# avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(20)]) / f1_denominator




# for i in range(20):
#     confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
#     precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

# confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(20)] )
# precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
# precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

# file.write('Test Set (Flores_romanized) : Precision-Recall-F1-Matrix')
# file.write('\n')
# for i in range(22):
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

# file_confusion_matrix = open('../result/confusion_matrix_ai4b_romanized.csv', 'w')
# confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

# for i in range(21):
#     confusion_matrix_csv_writer.writerow(confusion_matrix[i])

# file_confusion_matrix.close()









# result_test = model.test("../corpus/test_combine_ai4b_romanized.txt")


# file.write('Test Set (ai4b_romanized)')
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
















# # inference for dakshina_filtered
# file_test = open('../../../../../../Dakshina/dakshina_filtered/test_combine_dakshina.txt', 'r')
# lines_test = file_test.read().split('\n')

# # save predictions
# file_predictions = open('../result/predictions_dakshina_filtered.csv', 'w')
# csv_writer_predictions = csv.writer(file_predictions)
# csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# file_predictions_right = open('../result/right_predictions_dakshina_filtered.csv', 'w')
# csv_writer_predictions_right = csv.writer(file_predictions_right)
# csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_wrong = open('../result/wrong_predictions_dakshina_filtered.csv', 'w')
# csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
# csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# # save confusion matrix
# confusion_matrix = []
# for i in range(20):
#     confusion_matrix.append( [0]*20 )
    
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
# for i in range(20):
#     precsison_recall_f1.append([0] * 3)

# precision_denominator = 0
# recall_denominator = 0
# f1_denominator = 0

# # Computing precision, recall and f1
# for i in range(20):
#     no_of_correctly_predicted = confusion_matrix[i][i]
#     total_predictions_as_i = 0

#     precision = 0
#     recall = 0
#     f1_value = 0

#     # true predicted i values out of all predicted i values
#     for j in range(20):
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

# avg_precision = sum([precsison_recall_f1[i][0] for i in range(20)]) / precision_denominator
# avg_recall = sum([precsison_recall_f1[i][1] for i in range(20)]) / recall_denominator
# avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(20)]) / f1_denominator




# for i in range(20):
#     confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
#     precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

# confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(20)] )
# precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
# precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

# file.write('Test Set (dakshina filtered) : Precision-Recall-F1-Matrix')
# file.write('\n')
# for i in range(22):
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

# file_confusion_matrix = open('../result/confusion_matrix_dakshina_filtered.csv', 'w')
# confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

# for i in range(21):
#     confusion_matrix_csv_writer.writerow(confusion_matrix[i])

# file_confusion_matrix.close()









# result_test = model.test("../../../../../../Dakshina/dakshina_filtered/test_combine_dakshina.txt")


# file.write('Test Set (dakshina filtered)')
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

























# # inference for dakshina_romanized
# file_test = open('../corpus/test_combine_dakshina_romanized.txt', 'r')
# lines_test = file_test.read().split('\n')

# # save predictions
# file_predictions = open('../result/predictions_dakshina_romanized.csv', 'w')
# csv_writer_predictions = csv.writer(file_predictions)
# csv_writer_predictions.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# file_predictions_right = open('../result/right_predictions_dakshina_romanized.csv', 'w')
# csv_writer_predictions_right = csv.writer(file_predictions_right)
# csv_writer_predictions_right.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )


# file_predictions_wrong = open('../result/wrong_predictions_dakshina_romanized.csv', 'w')
# csv_writer_predictions_wrong = csv.writer(file_predictions_wrong)
# csv_writer_predictions_wrong.writerow( [ 'Sentence', 'Ground truth', 'Prediction', 'Score' ] )



# # save confusion matrix
# confusion_matrix = []
# for i in range(20):
#     confusion_matrix.append( [0]*20 )
    
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
# for i in range(20):
#     precsison_recall_f1.append([0] * 3)

# precision_denominator = 0
# recall_denominator = 0
# f1_denominator = 0

# # Computing precision, recall and f1
# for i in range(20):
#     no_of_correctly_predicted = confusion_matrix[i][i]
#     total_predictions_as_i = 0

#     precision = 0
#     recall = 0
#     f1_value = 0

#     # true predicted i values out of all predicted i values
#     for j in range(20):
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

# avg_precision = sum([precsison_recall_f1[i][0] for i in range(20)]) / precision_denominator
# avg_recall = sum([precsison_recall_f1[i][1] for i in range(20)]) / recall_denominator
# avg_f1_score = sum([precsison_recall_f1[i][2] for i in range(20)]) / f1_denominator




# for i in range(20):
#     confusion_matrix[i].insert(0, confusion_matrix_reverse_mapping[i] ) 
#     precsison_recall_f1[i].insert(0, confusion_matrix_reverse_mapping[i])

# confusion_matrix.insert( 0, [''] + [confusion_matrix_reverse_mapping[i] for i in range(20)] )
# precsison_recall_f1.insert( 0, ['', 'precision', 'recall', 'f1'])
# precsison_recall_f1.append( ['Avg', avg_precision, avg_recall, avg_f1_score] )

# file.write('Test Set (dakshina romanized) : Precision-Recall-F1-Matrix')
# file.write('\n')
# for i in range(22):
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

# file_confusion_matrix = open('../result/confusion_matrix_dakshina_romanized.csv', 'w')
# confusion_matrix_csv_writer = csv.writer(file_confusion_matrix)

# for i in range(21):
#     confusion_matrix_csv_writer.writerow(confusion_matrix[i])

# file_confusion_matrix.close()









# result_test = model.test("../corpus/test_combine_dakshina_romanized.txt")


# file.write('Test Set (dakshina romanized)')
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


