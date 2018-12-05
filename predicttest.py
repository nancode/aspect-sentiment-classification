import pandas as pd
import classmodel as trainmodel


file_no = 2
print('Metrics for Test Dataset %s' % file_no)
dict = trainmodel.model('./data-%s_train.csv' % file_no)
data = pd.read_csv('./Data-%s_test.csv'% file_no, header=0)

testData = trainmodel.preprocess(data)
vect = dict['featurevector']
classifier = dict['classifier']
featurevect = vect.transform(testData)
predictions = classifier.predict(featurevect)

#print(predictions)

text_id=data['example_id']
f=open("Unaiza_Faiz_VijayaNandhini_Sivaswamy_Data-%s.txt" % file_no,"w")
#newFrame=pd.DataFrame({'example_id':ids,
for i in range(len(text_id)):
    #print(predictions[i])
    #print(text_id[i]+";;" +str(predictions[i])+"\n")
    f.write(text_id[i]+";;" +str(predictions[i])+"\n")