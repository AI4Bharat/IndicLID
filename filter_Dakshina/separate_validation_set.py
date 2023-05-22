lang_code_list = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'ta', 'te', 'ur']

for lang_code in lang_code_list:
    f_org = open('scored_lang_wise_org_files/'+lang_code+'_scored.csv', 'r')
    lines_org = f_org.read().split('\n')[1:]
    f_org.close()
    
    lines_org = [line for line in lines_org if line]

    lines_validation = []
    lines_non_validation = []

    for line in lines_org:
        length = line.split(',')[-1]
        pred_score = line.split(',')[-2]
        predicted_label = line.split(',')[-3]
        ground_truth = line.split(',')[-4]

        flag = 0
        
        if predicted_label != ground_truth:
            flag = 1
        
        if float(pred_score) < 0.8:
            flag = 1
        
        if int(length) < 6:
            flag = 1
        
        if flag == 1:
            lines_validation.append(line)
        else:
            lines_non_validation.append(line)
    
    f_valid = open('validation_set/'+lang_code+'_scored.csv', 'w')
    f_valid.write('\n'.join(lines_validation))
    f_valid.close()

    f_non_valid = open('non_validation_set/'+lang_code+'_scored.csv', 'w')
    f_non_valid.write('\n'.join(lines_non_validation))
    f_non_valid.close()
