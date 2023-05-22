import pandas as pd
lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'ta', 'te', 'ur']
# lang_code_list = ['gu']

final_merge_xls = pd.ExcelFile('Final_merged_all.xlsx')

file = open('filter_lang_stats_native.txt', 'w')
file.write( 'lang_code' + '\t' + 'length_native_sen_non_valid' + '\t' + 'length_native_sen_valid' + '\t' + 'length_native_dakshina_filter_list' + '\t' + 'length_native_sen_valid_minus_dakshina_filter_list'+ '\t' + 'length_final_merge_list' + '\n')


for lang_code in lang_code_list:

    # validated native inputs
    df = pd.read_excel(final_merge_xls, lang_code + '_scored')
    dakshina_filter_list = df.values.tolist()
    # print(dakshina_filter_list[0:10])
    native_dakshina_filter_list = [line[0] for line in dakshina_filter_list if line[1] == 0 ]
    # print(native_dakshina_filter_list[0])

    # non vlidation set
    df_non_valid = pd.read_csv('non_validation_set/'+lang_code+'_scored.csv', header = None)
    lines_non_validation = df_non_valid.values.tolist()
    native_sen_non_valid = [line[0] for line in lines_non_validation]
    # print(native_sen_non_valid[0])

    native_final_filter_list = native_dakshina_filter_list + native_sen_non_valid

    # for reverificaiton open validaiton set
    df_valid = pd.read_csv('validation_set/'+lang_code+'_scored.csv', header = None)
    lines_validation = df_valid.values.tolist()
    native_sen_valid = [line[0] for line in lines_validation]
    # print(native_sen_valid[0])

    # reverificaiton
    set_difference = set(native_sen_valid).difference(set(native_dakshina_filter_list))
    
    file.write( lang_code + '\t' + str(len(native_sen_non_valid)) + '\t' + str(len(native_sen_valid)) + '\t' + str(len(native_dakshina_filter_list)) + '\t' + str(len(set_difference))+ '\t' + str(len(native_final_filter_list)) + '\n')

    print(lang_code + " " + str(len(set_difference)))
    # print(set_difference)
    
    file_dakshina_filter = open('final_merge_native_set/'+lang_code+'_filter.txt', 'w')
    file_dakshina_filter.write('\n'.join(native_final_filter_list))
    file_dakshina_filter.close()

file.close()


    # f_org = open('scored_lang_wise_org_files/'+lang_code+'_scored.csv', 'r')
    # lines_org = f_org.read().split('\n')[1:]
    # f_org.close()
    
    # lines_org = [line for line in lines_org if line]

    # lines_validation = []
    # lines_non_validation = []

    # for line in lines_org:
    #     length = line.split(',')[-1]
    #     pred_score = line.split(',')[-2]
    #     predicted_label = line.split(',')[-3]
    #     ground_truth = line.split(',')[-4]

    #     flag = 0
        
    #     if predicted_label != ground_truth:
    #         flag = 1
        
    #     if float(pred_score) < 0.8:
    #         flag = 1
        
    #     if int(length) < 6:
    #         flag = 1
        
    #     if flag == 1:
    #         lines_validation.append(line)
    #     else:
    #         lines_non_validation.append(line)
    
    # f_valid = open('validation_set/'+lang_code+'_scored.csv', 'w')
    # f_valid.write('\n'.join(lines_validation))
    # f_valid.close()

    # f_non_valid = open('non_validation_set/'+lang_code+'_scored.csv', 'w')
    # f_non_valid.write('\n'.join(lines_non_validation))
    # f_non_valid.close()