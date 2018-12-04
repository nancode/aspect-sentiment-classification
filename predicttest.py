import pandas as pd
import classmodel as trainmodel

dict = trainmodel.model('./data-1_train.csv')

data = pd.read_csv('./Data-1_test.csv', header=0)

testData = trainmodel.preprocess(data)
vect = dict['featurevector']
classifier = dict['classifier']
featurevect = vect.transform(testData)
predictions = classifier.predict(featurevect)
#print(predictions)

text_id=data['example_id']
f=open("Unaiza_Faiz_VijayaNandhini_Sivaswamy_Data-1.txt","w")
#newFrame=pd.DataFrame({'example_id':ids,
for i in range(len(text_id)):
    #print(predictions[i])
    #print(text_id[i]+";;" +str(predictions[i])+"\n")
    f.write(text_id[i]+";;" +str(predictions[i])+"\n")