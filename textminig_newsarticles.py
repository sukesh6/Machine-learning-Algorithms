import nltk
nltk.download('punkt')

#pip install lxml_html_clean
#pip install newspaper3k 
import newspaper  # Importing the newspaper library for web scraping
from newspaper import Article # Importing the Article class for article extraction
import pandas as pd
from textblob import  TextBlob
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt # Importing the matplotlib library for visualization
from wordcloud import WordCloud  # Importing the WordCloud class for generating word clouds
import os 
import warnings
import re # Importing the re module for regular expressions


url1="article_url"
article1=Article(url1)


url2="article_url0"
article2=Article(url2)


url3="article_url"
article3=Article(url3)


url4="article_url"
article4=Article(url4)


url5="article_url"
article5=Article(url5)


url6="article_url"
article6=Article(url6)


url7="article_url"
article7=Article(url7)


url8="article_url"
article8=Article(url8)


url9="article_url"
article9=Article(url9)


url10='article_url'
article10=Article(url10)


BF7_Articles = [article1, article2,article3,article4,article5,article6,article7,article8,article9,article10]
for x in BF7_Articles:
    x.download()
    x.parse()
    x.nlp()
    
Summary1=TextBlob(article1.summary)
Summary2=TextBlob(article2.summary)
Summary3=TextBlob(article3.summary)
Summary4=TextBlob(article4.summary)
Summary5=TextBlob(article5.summary)
Summary6=TextBlob(article6.summary)
Summary7=TextBlob(article7.summary)
Summary8=TextBlob(article8.summary)
Summary9=TextBlob(article9.summary)
Summary10=TextBlob(article10.summary)

#Sentiment Analysis
''' Polarity: Polarity refers to the sentiment expressed in the text. It typically indicates whether the sentiment is positive, negative, or neutral.
range (-1 to +1) -1 is negative
                 +1 is positive
                  0 is neutral
Subjectivity: Subjectivity indicates how much a text expresses personal opinions or feelings as opposed to objective facts. It measures the degree to which a text is opinionated.
range (0 to 1) 0 is objective text (fact based)
               1 is highle subjective text (opinion based) '''
print(Summary1.sentiment) 
print(Summary2.sentiment)
print(Summary3.sentiment)
print(Summary4.sentiment)
print(Summary5.sentiment)
print(Summary6.sentiment)
print(Summary7.sentiment)
print(Summary8.sentiment)
print(Summary9.sentiment)
print(Summary10.sentiment)

# initialize list of lists
data = [Summary1.sentiment,Summary2.sentiment,Summary3.sentiment,Summary4.sentiment,Summary5.sentiment,Summary6.sentiment,Summary7.sentiment,Summary8.sentiment,Summary9.sentiment,Summary10.sentiment]
# pandas DataFrame
df = pd.DataFrame(data, columns=['Polarity', 'Subjectivity'],index=['Zee News','NDTV','India','Time of India','IndiaExpress','LiveMint','EconomicTimes','Onmanorama','News18','TheHindu'])
 
# print dataframe.
df

#Graphical Representation

plt.figure(figsize=(18,9))
data1 = [Summary1.sentiment[0],Summary2.sentiment[0],Summary3.sentiment[0],Summary4.sentiment[0],Summary5.sentiment[0],Summary6.sentiment[0],Summary7.sentiment[0],Summary8.sentiment[0],Summary9.sentiment[0],Summary10.sentiment[0]]
data2 = [Summary1.sentiment[1],Summary2.sentiment[1],Summary3.sentiment[1],Summary4.sentiment[1],Summary5.sentiment[1],Summary6.sentiment[1],Summary7.sentiment[1],Summary8.sentiment[1],Summary9.sentiment[1],Summary10.sentiment[1]]
width =0.3
leg=['Polarity', 'Subjectivity']
labels = ['article 1','article 2','article 3','article 4','article 5','article 6','article 7','article 8','article 9','article 10']
plt.bar(labels, data1, width=width,color='r')
plt.bar(np.arange(len(data2))+ width, data2, width=width,color='blue')
plt.xlabel('Different News Channels', fontsize=14)
plt.ylabel('Polarity & Subjectivity', fontsize=14)
plt.title('Sentiment Analysis on News Articles of BF-7 Covid Variant',fontsize=18)
plt.grid()
plt.legend(leg,loc=2,fontsize=15)
plt.show()


