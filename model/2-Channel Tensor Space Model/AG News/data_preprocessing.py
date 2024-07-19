import pandas as pd
import os
import string
import re

def exclude_list(input_list, cnt):
    return input_list[:cnt*2000]+input_list[cnt*2000+2000:8000]+input_list[8000:cnt*2000+8000]+input_list[cnt*2000+10000:]

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def clean_text(text):
    text=str(text).lower() #Converts text to lowercase
    text=re.sub('\d+', '', text) #removes numbers
    text=re.sub('\[.*?\]', '', text) #removes HTML tags
    text=re.sub('https?://\S+|www\.\S+', '', text) #removes url
    text=re.sub(r"["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", "", text) #removes emojis
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text) #removes punctuations
    return text

 #텍스트 큐보이드 생성을 위한 인덱스
def train_idx_list(cnt):
    original_train_lst=[i for i in range(16000)]
    return original_train_lst[:cnt*2000]+original_train_lst[cnt*2000+2000:8000]+original_train_lst[8000:cnt*2000+8000]+original_train_lst[cnt*2000+10000:]

def test_idx_list(cnt):
    original_train_lst=[i for i in range(16000)]
    return original_train_lst[cnt*2000:cnt*2000+2000]+original_train_lst[cnt*2000+8000:cnt*2000+10000]


#문서 길이를 조정하는 코드 텍스트 리스트와 최대 길이를 입력하면 단어 시퀀스를 최대 길이 이내로 truncate 해줌
def limit_words(text_list, max_words):
    new_list = []
    for text in text_list:
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
        new_text = ' '.join(words)
        new_list.append(new_text)
    return new_list

#AG 뉴스 데이터를 불러오고 타이틀과 본문 텍스트를 결합
def combine_title_and_description(df):
    # Returns a dataset with the title and description fields combined
    df['text'] = df[['Title', 'Description']].agg('. '.join, axis=1)
    df = df.drop(['Title', 'Description'], axis=1)
    return df