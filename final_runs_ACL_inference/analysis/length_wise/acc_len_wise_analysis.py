import json
import random
import matplotlib.pyplot as plt

length_bucket = 3
max_sen_len = 30

file = open('../result/dict_predictions.json', 'r')
dict_predictions_org = json.loads(file.read())

dict_predictions = {}
for key in dict_predictions_org:
    if int(key)<=max_sen_len:
        dict_predictions[int(key)] = dict_predictions_org[key]

# soretd_temp = list(dict_predictions.keys())
# soretd_temp.sort()
# for key in soretd_temp:
#     print(key, len(dict_predictions[key]))

max_len = 0
for keys in dict_predictions.keys():
    if max_len<keys:
        max_len=keys      

print("max_len : ", max_len)
print("dict_predictions", len(dict_predictions))
i = 1
k = 1
dict_predictions_bucket = {}
while i<=max_len:
    
    if i in dict_predictions:
        if k in dict_predictions_bucket:
            dict_predictions_bucket[k] += dict_predictions[i]
        else:
            dict_predictions_bucket[k] = dict_predictions[i]
    i+=1
    if i%length_bucket==1:
        k+=1
print("dict_predictions_bucket : ",len(dict_predictions_bucket))
print("dict_predictions_bucket : ",dict_predictions_bucket.keys())

min_samples = len(dict_predictions_bucket[list(dict_predictions_bucket.keys())[0]])
for key in dict_predictions_bucket:
    if min_samples > len(dict_predictions_bucket[key]):
        min_samples = len(dict_predictions_bucket[key])
print("min_samples ", min_samples)

dict_samples = {}
for key in dict_predictions_bucket:
    dict_samples[key] = random.choices(dict_predictions_bucket[key], k = min_samples)

print("dict_samples : ",len(dict_samples))

dict_accuracy = {}
for key in dict_samples:
    count = 0
    for i in dict_samples[key]:
        if i[1]==1:
            count+=1
        
    dict_accuracy[key] = count/len(dict_samples[key])

print("dict_accuracy : ",len(dict_accuracy))

# with open("../result/dict_accuracy.json", "w") as outfile:
#     json.dump(dict_accuracy, outfile)

xpoints = []
ypoints = []
for key in dict_accuracy:
    xpoints.append(key*length_bucket)
    ypoints.append(dict_accuracy[key]*100)

plt.xlabel("Length of the sentence (words)")
plt.ylabel("Accuracy")

plt.axis([0, max_sen_len+length_bucket, 40, 105])
plt.plot(xpoints, ypoints)
plt.savefig('../result/len_wise_accuracy.png')
