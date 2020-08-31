from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import docx2txt
import re
import jieba
import numpy as np
import pandas as pd
import docx
import os
from tqdm import tqdm
from pytorch_class import Classifier
import collections


jieba.case_sensitive = True
jieba.set_dictionary('jieba_dict/dict.txt.big.txt')
jieba.load_userdict('jieba_dict/tableName.txt')
jieba.load_userdict('jieba_dict/programNameList.txt')
jieba.load_userdict('jieba_dict/dict.txt')
jieba.load_userdict('jieba_dict/allvocb.txt')

stopWordFile = open('jieba_dict/stop_word.txt', 'r', encoding='UTF-8')
stopWordList = stopWordFile.read().split('\n')
tableNameFile = open('jieba_dict/tableName.txt', 'r', encoding='UTF-8')
tableNameList = tableNameFile.read().split('\n')
programNameFile = open('jieba_dict/programNameList.txt', 'r', encoding='UTF-8')
programNameList = programNameFile.read().split('\n')


def keywords_extraction(all_content):
    spec_vocab = load_vocab('jieba_dict/spec_vocab.txt')
    k_list = spec_vocab.keys()
    kw_string_list = []
    for each in all_content:
        tokens = each.split(' ')
        kw_string = ''
        for token in tokens:
            if token in k_list:
                kw_string = kw_string+token+' '
        kw_string_list.append(kw_string)
    return kw_string_list

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file,'r', encoding='utf-8') as f:
        line = f.readline().strip('\n')
        while line:
            vocab['{}'.format(line)] = index
            index = index+1
            line  = f.readline().strip('\n')
    return vocab

def tfidf(specs):
    index = specs
    path  = './dataset/spec/'
    all_content = []
    for spec in tqdm(specs):
        try:
            document = docx2txt.process(path+spec)
            content = ' '.join(re.split(r'[\n\t]',document))
        except:
            print(spec)
        all_content.append(content)
    
    kw_string_list = keywords_extraction(all_content)

    #將停詞引入模型,tfidf=TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=stopword)
    vector=TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=["\n"])
    tfidf=vector.fit_transform(kw_string_list)#模型向量化
    keywords = vector.get_feature_names()
    df_tfidf = pd.DataFrame(tfidf.toarray(),columns=keywords, index=index)
    
    count_vector = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=["\n"])  
    word_frequency = count_vector.fit_transform(kw_string_list)
    all_words = count_vector.get_feature_names()
    word_fr_df = pd.DataFrame(word_frequency.toarray(),columns=all_words, index=index)

    return df_tfidf, kw_string_list, keywords

# 將使用者上傳之文件取出tfidf向量
def tfidf_user_upload(specs, kw_string_list, dataset_key_word):
    document = docx2txt.process(specs)
    content = ' '.join(re.split(r'[\n\t]',document))
    userupload_kw_string = keywords_extraction([content])

    userupload_kw_list = userupload_kw_string[0].split(" ")

    userupload_kw_filter = []
    for i in range(len(userupload_kw_list)):
        if userupload_kw_list[i].lower() in dataset_key_word:
            userupload_kw_filter.append(userupload_kw_list[i])

    print(userupload_kw_filter)
    kw_string_list.append(" ".join(userupload_kw_filter))

    vector = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=["\n"])
    tfidf = vector.fit_transform(kw_string_list)
    tfidfArray = tfidf.toarray()

    return tfidfArray[-1]

# 將資料集降維成128維
def PCA_dimension_reduction(df_tfidf, specs):
    pca = PCA(n_components = 128)
    pcaModel = pca.fit(df_tfidf)
    newData = pca.transform(df_tfidf)
    newData_df = pd.DataFrame(newData,index=specs)
    return newData_df, newData, pcaModel

# 將使用者上傳的資料降維成128維
def PCA_dimension_reduction_user_upload(user_tfidf_vect, pca):
    user_tfidf_vect = user_tfidf_vect.reshape(1,-1)
    dimensionReduction = pca.transform(user_tfidf_vect)

    return dimensionReduction[0]

# 將使用者的tfidf pca向量與資料集其他spec的向量合併
def concat_user_dataset_V(user_pca_vect, dataset_pca_vect):
    dataset_pca_vect = dataset_pca_vect.tolist()
    for i in range(len(dataset_pca_vect)):
        dataset_pca_vect[i].extend(user_pca_vect)
    return np.array(dataset_pca_vect)

# 預測單筆資料是否相關
def predict(x):
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        model = Classifier()
        model.load_state_dict(torch.load('model/classifier_0.35_785.pth'))
        model.eval()
        output = F.softmax(model(x))
        _, pred = torch.max(output, 0)
    return (output,pred)

# 將一筆筆資料預測並產生預測結果陣列，也加上對應到的文件檔名
def predict_loop(prepared_data, labelSpecList):
    outputList = []
    for i in range(len(prepared_data)):
        result = predict(prepared_data[i])
        pro = result[0].tolist()[1]
        predict_label = result[1].item()
        docx_name = labelSpecList[i]
        outputList.append([docx_name, predict_label, pro])
    
    return outputList

# 將預測結果依照probability排序(由大到小)
def sort_predicted_list(predicted_list):
    predicted_list.sort(reverse=True, key = lambda s: s[2])
    return predicted_list
