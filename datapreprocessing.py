import re
import nltk


def cleanStatement(text):
    cleanedText = []
    for each in text:
        #if(each=="casing"):
          #  print(each)
        cleanText = each
        #print (cleanText)
        cleanText = cleanText.lower() #convert text to lower case
        #print (cleanText)
        cleanText = re.sub(r'[?;!$:+*"\']*','',cleanText) #remove punctuations
        #print (cleanText)
        cleanText = re.sub(r'(\[comma\])',' ',cleanText) #remove [comma]
        #print (cleanText)
        cleanText = re.sub(r'[./()\-=_]',' ',cleanText)

        #print (cleanText)
        #remove stop words
        from nltk.corpus import stopwords
        stopwordsSet = stopwords.words('english')
        #print(stopwordsSet)
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(cleanText)
        cleanText=''
        for word in tokens:
            if not word in stopwordsSet:
                cleanText+=word+' '
        #print(tokens)

        #print(cleanText)

        #lemmatizing the text

        from nltk.stem import WordNetLemmatizer
        tokens =word_tokenize(cleanText)
        cleanText =''
        for word in tokens:
            cleanText+= WordNetLemmatizer().lemmatize(word)+' '
        #print(cleanText)

        #Stemming - PorterStemmer
        #from nltk.stem.porter import PorterStemmer
        #tokens =word_tokenize(cleanText)
        #cleanText =''
        #for word in tokens:

        #    cleanText+= PorterStemmer().stem(word)+' '

        #if(each=="casing"):
           # print(cleanText)
        #print(cleanText)



        cleanedText.append(cleanText)
    return cleanedText

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results


def getContextWindow(cleanedText, cleanedAspect):
    j=0
    textContextList = []
    for text, aspect in zip(cleanedText,cleanedAspect):
        j=j+1
        textContext=[]

        from nltk.tokenize import word_tokenize
        texttokens = word_tokenize(text)
        aspectToken = word_tokenize(aspect)
        #print("Text: "+text)
        #print("Aspect: "+aspect)
        results = find_sub_list(aspectToken,texttokens)
        if(len(results)==0):
            #print(j)
            #print("Text:"+text)
            #print ("Aspect:"+aspect)
            textContext = text
            #print(textContext)
        else:
            #textContext = []
            startIndex =0
            if(results[0][0]<3):
                startIndex = 0
            else:
                startIndex = results[0][0]-3
            endIndex = 0
            if(results[0][1]>len(texttokens)-3):
                endIndex = len(texttokens)
            else:
                endIndex = results[0][1]+3
            for i in range(startIndex,endIndex):
                textContext.append(texttokens[i])

            #print(textContext)
        #print(results[0][0])
        textContextList.append(' '.join(textContext))

    return textContextList


