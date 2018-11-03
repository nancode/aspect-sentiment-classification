import re
import nltk

def cleanStatement(text):
    cleanedText = []
    for each in text:
        cleanText = each
        cleanText = cleanText.lower() #convert text to lower case
        cleanText = re.sub(r'[.?;!$()_:+"*-]*','',cleanText) #remove punctuations
        cleanText = re.sub(r'(\[comma\])','',cleanText) #remove [comma]

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

        print(cleanText)

        #lemmatizing the text

        from nltk.stem import WordNetLemmatizer
        tokens =word_tokenize(cleanText)
        cleanText =''
        for word in tokens:
            cleanText+= WordNetLemmatizer().lemmatize(word)+' '
        print(cleanText)

        #Stemming - PorterStemmer
        #from nltk.stem.porter import PorterStemmer
        #tokens =word_tokenize(cleanText)
        #cleanText =''
        #for word in tokens:
        #    cleanText+= PorterStemmer().stem(word)+' '
        #print(cleanText)



        cleanedText.append(cleanText)
    return cleanedText

cleanStatement({"Yeah[comma] of course smarty pants \"fix it now\")Software - Compared to the early 2011 edition I did see inbuilt applications crashing and it prompted me to send the report to Apple (which I promptly did)."})
cleanStatement({"inbuilt applications"})
