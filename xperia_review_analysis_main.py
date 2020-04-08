#!/usr/bin/env python
# coding: utf-8

# ### The purpose of the analysis is to explore the customer satisfaction toward different Xperia handsets which are scraped from Amazon product review pages (the scraper script can be found in my the other notebook), and to build a simple prediction model to predict whether a product review is positive/neural/negative

# In[1]:


import pandas as pd # for data analysis
import numpy as np # for numerical analysis
import matplotlib.pyplot as plt # plot charts
from matplotlib import style
import seaborn as sns # different chart ploting module
import re # regex (regular expression) 
from __future__ import absolute_import, division, print_function
# to display charts in jupyter notebook


# ### Define column name and import the data

# In[118]:


columns = ['product_name','price','author','date','rating','summary','review']
xperia_reviews_df = pd.read_csv('xperia_product_reviews.txt',sep='|',names=columns,error_bad_lines=True)
xperia_reviews_df.head() # view top 5 rows 


# ### Check the column data format and any missing value

# In[119]:


xperia_reviews_df.info()


# #### There are 13,360 rows of data and there is no missing value. Apart from rating, the rest of the fields are string format. For product level analysis, the product name need to be extracted from the product full name. Besides, the produce value and date need converting into value format and date format accordingly.

# ### Alternative way to check missing value visually

# In[132]:


sns.heatmap(xperia_reviews_df.isnull(),yticklabels=False,cbar=False,cmap='RdYlGn')


# In[4]:


from dateutil.parser import parse # converting date from string to date format


# In[121]:


xperia_reviews_df.product_name.unique()


# #### It appears that apart from compact and SONY 1303-3116 Xperia handsets, its safe to extract the first 3 words from the full product name to form the shortened produce name.

# In[122]:


def extract_name(name):
    if 'compact' in name.lower() or 'SONY 1303-3116 Xperia' in name:
        return ' '.join(name.split()[:4])
    else:
        return ' '.join(name.split()[:3])

xperia_reviews_df['product_name_modified'] = xperia_reviews_df.product_name.apply(extract_name)


# In[123]:


xperia_reviews_df['review_date'] = xperia_reviews_df.date.apply(lambda x : parse(x.replace('on ','')))
xperia_reviews_df['product_value']= pd.to_numeric(xperia_reviews_df.price.apply(lambda x:(x.split('-')[0].strip().replace('£',''))))


# # 

# In[124]:


xperia_reviews_df.head()


# In[125]:


xperia_reviews_df.product_name_modified.value_counts().plot(kind='barh',title = 'Review count by product'
                                                            ,figsize=(15,5))


# ### Let's check the overall rating distribution

# In[126]:


pd.DataFrame(xperia_reviews_df.rating.value_counts().sort_index()/xperia_reviews_df.shape[0]).plot(kind='bar')


# #### It appears that the reviews are very polarized 

# In[127]:


ave_rating = xperia_reviews_df.rating.sum()/xperia_reviews_df.shape[0]
print 'average rating : {}'.format(ave_rating)


# In[128]:


price_rating = xperia_reviews_df.groupby(['product_value']).agg({'review':'count','rating':'sum'}).reset_index()
price_rating['ave._rating'] = price_rating.rating/price_rating.review
price_rating.plot(x='product_value',y='ave._rating',kind='bar',figsize=(15,5))

plt.axhline(y=ave_rating, color='r', linestyle='-')
plt.title('Average rating by price range')


# #### Sony mobile has very diverted product range from value to mid to premium products 
# #### It appears that the product range with a price £280 onward tend to have a higher average rating than the overall rating (red line) 
# #### But there is an odd price point 6.99 and it turns out these are the phone handset case rather than the actual handsets. Let's remove them from the dataset

# In[15]:


xperia_reviews_df[xperia_reviews_df.product_value==6.99].product_name.unique()


# In[16]:


xperia_reviews_df = xperia_reviews_df[xperia_reviews_df.product_value>6.99]


# ### Calculating the average rating by product

# In[17]:


rating_by_product = xperia_reviews_df.groupby('product_name_modified').agg({'review':'count','rating':'sum'})
rating_by_product


# In[105]:


rating_by_product['average_rating'] = rating_by_product.rating / rating_by_product.review
rating_by_product.average_rating.sort_values(ascending = True).plot(kind='barh',figsize=(15,5)
                                                                    ,title='Average rating by product')


# #### The Xperia X family product perform better than the other handsets, except the Xperia XA.
# #### Interestingly there are some value line handsets have a high average rating espcially Xperia E4.
# #### Instead of looking at the average rating, its also important to investigate the distribution of the rating for each product

# In[106]:


rating_table = pd.pivot_table(data = xperia_reviews_df, index = 'product_name_modified'
               ,columns= 'rating'
               ,values = 'review'
              ,aggfunc=len)


# In[107]:


rating_table2.sort_values(by=5.0).drop('total',axis = 1).plot(kind='barh',stacked = True,cmap='RdYlGn',figsize=(20,10))


# #### Not only the Xperia X handsets have a high proportion of highly satisfied customers but more importantly they have a low proportion of unsatisfied customers
# #### On the other spectrum, M4 Aqua has a very large portion of unsatisfied customers
# 
# #### Let's get a bit fancy with the wordcloud and try to get a general view what the customers talked about the E4 and M4 Aqua handset

# In[19]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS) | {'phone','sony','xperia','E4','SIM','free','white','black','smartphone'}


# In[20]:


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
#         max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))
    
    fig = plt.figure(1, figsize=(15, 10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    


# ### The below are really cool, but maybe give them labels (even though its defined in the code, i found it hard to follow first time round)

# In[26]:


show_wordcloud(xperia_reviews_df[xperia_reviews_df.product_name_modified =='Sony Xperia E4']['review'])


# ### It seems there are a lot of postive conversations around price, battery, camera, easy (ease of use?) for the E4 handset

# In[110]:


show_wordcloud(xperia_reviews_df[(xperia_reviews_df.product_name_modified =='Sony M4 Aqua')&(xperia_reviews_df.rating==1)]['review'])


# ### The 8GB storage memory issue in M4 Aqua really stand out

# In[24]:


show_wordcloud(xperia_reviews_df[xperia_reviews_df.rating >4]['review'])


# In[112]:


show_wordcloud(xperia_reviews_df[xperia_reviews_df.rating <=2]['review'])


# ### Overall areas such as screen, picture, size, style, design has scored low rating from the customers. Let's dive into the screen related topic to get a general view

# In[23]:


show_wordcloud(xperia_reviews_df2[(xperia_reviews_df2.review.str.contains('screen'))&(xperia_reviews_df2.rating_star==1)].review)


# #### The issue of screen crack is the dominated topic

# # Simple Sentiment Model
# ### Since we have the review and rating from the customers, we can use this data to build a sentiment model. But there are some data preperations need to do before building the model:
# ### 1. Tokenize the sentence (Breaking down the sentence to each individual word; but I used split function instead of words tokenizing from nltk for the time being)
# ### 2. Remove punctuation, stopwords (common words) and numbers
# ### 3. Stemming the words (remove morphological affixes from words, leaving only the word stem)
# 
# ### After that, we split the data into training and testing datasets. Then extracting the text features from both datasets using the TfidfVectorizer function. Last, we feed the text features as the input variables to train the model. I also narrowed down the 5 target variables into 3, with rating 1/ 2 are classified as 0, 3 as 1 and 4/5 as 2 to represent the negative/neural/positive sentiment

# In[28]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[30]:


import nltk
from nltk.stem.porter import PorterStemmer


# In[31]:


stemer = PorterStemmer()


# In[32]:


def cleanup(sentence):
    sentence =  re.sub(r'[^\w\s]','',sentence)
    return re.sub('[^a-z]+','',sentence)

def tokenizing(sentence):
#     tokens = nltk.word_tokenize(sentence)
    tokens = sentence.split()
    stemed_tokens = []
    for token in tokens:
        if token.lower().strip() not in list(stopwords):
            stemed_tokens.append(stemer.stem(cleanup(token)))
    return ' '.join(stemed_tokens)


# In[33]:


# tokenizing function demo
sentence = 'I have eaten 2 10cm long sandwiches'
tokenizing(sentence)


# In[34]:


xperia_reviews_df['cleaned_review'] = xperia_reviews_df.review.apply(tokenizing)


# In[35]:


def sentiment_class(rating):
    if rating <=2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


# In[36]:


xperia_reviews_df['sentiment'] = xperia_reviews_df.rating.apply(sentiment_class)


# In[38]:


X_raw = xperia_reviews_df.cleaned_review
y=xperia_reviews_df.sentiment


# In[39]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[134]:


X_train_raw,X_test_raw,y_train,y_test = train_test_split(X_raw,y,test_size=0.3,random_state = 101)


# In[135]:


tfidf = TfidfVectorizer(ngram_range=(1,2))

X_train = tfidf.fit_transform(X_train_raw)
X_test = tfidf.transform(X_test_raw)


# ### Training a logistic regression model 

# In[151]:


lg = LogisticRegression()
lg.fit(X_train,y_train)
prediction = lg.predict(X_test)
print "Logistic Regression Model accuracy : {}".format(accuracy_score(y_test,prediction))


# ### Training a random forest model

# In[150]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)
prediction = rf.predict(X_test)
print "Random Forest Model accuracy : {}".format(accuracy_score(y_test,prediction))


# ### Check the actual count between actual Vs prediciton class

# In[138]:


confusion_matrix(y_test,prediction)


# ### Random forest model outperforms the logistic regression model, lets check the top 10 important variables from the winner model

# In[45]:


feature_df = pd.DataFrame(rf.feature_importances_.tolist(),tfidf.get_feature_names())
feature_df.reset_index(inplace=True)
columns = ['feature','importance']
feature_df.columns = columns


# In[114]:


feature_df.sort_values(by='importance',ascending=False).head(10)


# ### |Trying out the model accuracy with some hand-written reviews

# In[115]:


test_corpus = pd.Series(['the phone screen is crap','good quality picture','poor speaker'])
text_token = test_corpus.apply(tokenizing)
input_text = tfidf.transform(pd.Series(text_token))
prediction_text = rf.predict(input_text)


# In[149]:


def sentiment_map(x):
    if x == 0:
        return 'Negative'
    elif x == 1:
        return 'Neural'
    else:
        return 'Positive'
    
pd.DataFrame(pd.Series(prediction_text).apply(sentiment_map).tolist(),test_corpus,columns=['Sentiment'])


# ### Further development:
# #### 1. Deploying the model live in a web browser
# #### 2. Improving model performance by tuning the model hyperparameters
# #### 3. Increasing the prediction accuracy with deep learning model such as RNN model

# In[ ]:




