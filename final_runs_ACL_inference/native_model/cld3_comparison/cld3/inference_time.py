import fasttext
import random
import time
import cld3

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

def predict(sen):

    # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
        # for line in lines_resampled:
            
            # inputs, labels = data[0], data[1]
            # labels = line.split(' ')[0]
            # labels = confusion_matrix_mapping[ labels[9:] ]

    result = cld3.get_language(sen)
    pred_label = result.language

    # if fasttext_pred_score > threshold:
    return  pred_label
        # else:
        #     word_embeddings = tokenizer(sen, return_tensors="pt", padding=True, truncation=True, max_length=512)   
        #     word_embeddings = word_embeddings.to(device)
        #     # labels = labels.to(device)

        #     outputs = model(word_embeddings['input_ids'], 
        #                 token_type_ids=word_embeddings['token_type_ids'], 
        #                 attention_mask=word_embeddings['attention_mask']
        #                 # ,labels=labels
        #                 )
        #     _, predicted = torch.max(outputs.logits, 1)
            
        #     return confusion_matrix_reverse_mapping[predicted.item()]
            
                # print(outputs.logits)
                # print(predicted)
            
            # for sen, label, pred_label, logit in zip(inputs, labels, predicted, outputs.logits):
                
            #     fasttext_prediction = fasttext_model.predict(sen)
            #     fasttext_pred_label = fasttext_prediction[0][0]
            #     fasttext_pred_score = fasttext_prediction[1][0]
                
            #     pred_score = logit[pred_label.item()].item()
                
            #     if fasttext_pred_score > threshold:
            #         if confusion_matrix_mapping[fasttext_pred_label[9:]] == label.item():
            #             final_count+=1
            #     else:
            #         if label.item() == pred_label.item():
            #             final_count+=1
            #         else:             

            #     final_n+=1

            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    #     acc = ((100 * correct) / total)

    # final_acc = ((100 * final_count) / final_n)
    # print(f'Accuracy of the network on the {total} test inputs: {final_acc} ')   
    # file.write('Test Set ('+file_name+') : ' + str(final_acc) + '\n')









def measure_inference_time(lines_sampled):
    
    start_time = time.time()
    
    for line in lines_sampled:
        inputs = ' '.join(line.split(' ')[1:])
        # print(predict(line))
        predict(inputs)

    end_time = time.time()

    inference_time = end_time - start_time
    return inference_time

# preparing samples
files_in = open('../corpus/test_combine_native_12.txt', 'r')
lines_in = files_in.read().split('\n')
files_in.close()


file = open('../common_result/inference_time.txt', 'w')
# dims = [4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
# for dim in dims:


inference_time_list = []
for _ in range(100):
    lines_sampled = random.sample( lines_in, k=100 )
    inference_time = measure_inference_time(lines_sampled)
    inference_time_list.append(inference_time)
average_inference_time = sum(inference_time_list)/len(inference_time_list)
file.write('Inference time : ' + str(average_inference_time) + '\n')
# print('Inference time for threshold ' + str(threshold) + ' : ' + str(average_inference_time) + '\n')

file.close()




    


# evaluate( 'test_combine', model)
# evaluate( 'test_dakshina_original_roman', model)
# evaluate( 'valid_dakshina_original_roman', model)
# evaluate( 'test_dakshina_filter_roman', model)
# evaluate( 'test_dakshina_indicxlit_romanized', model)
# evaluate( 'test_combine_flores200_romanized', model)
# evaluate( 'test_combine_ai4b_romanized', model)




