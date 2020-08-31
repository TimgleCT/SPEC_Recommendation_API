import docx2txt
import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from tqdm import tqdm
import re
import os

jieba.case_sensitive = True
jieba.set_dictionary('jieba_dict/dict.txt.big.txt')
jieba.load_userdict('jieba_dict/tableName.txt')
jieba.load_userdict('jieba_dict/programNameList.txt')
jieba.load_userdict('jieba_dict/allvocb.txt')

stopWordFile = open('jieba_dict/stop_word.txt', 'r', encoding='UTF-8')
stopWordList = stopWordFile.read().split('\n')
tableNameFile = open('jieba_dict/tableName.txt', 'r', encoding='UTF-8')
tableNameList = tableNameFile.read().split('\n')
programNameFile = open('jieba_dict/programNameList.txt', 'r', encoding='UTF-8')
programNameList = programNameFile.read().split('\n')


# 將既有全部資料集做前處理
def preprocessing_allSPEC(source_directory):
    findEnNum = re.compile("[^A-Za-z0-9]")
    rule = re.compile("[^\u4e00-\u9fa5]")
    docList = []
    docIndex = []

    for file in tqdm(os.listdir(source_directory)):
        noStopWordTokenList = []
        if file.endswith(".docx"):
            filename = source_directory + file
            docIndex.append(file)
            docxToText = docx2txt.process(filename)

            docxToTextEnNum = "".join(findEnNum.sub('',docxToText))
            docxToText = "".join(rule.sub('',docxToText))

            tokenizeStr = " ".join(jieba.cut(docxToText))
            tokenizeStrEnNum = " ".join(jieba.cut(docxToTextEnNum))

            newTokenStr = tokenizeStr.split(" ")
            newTokenStrEnNum = tokenizeStrEnNum.split(" ")


            for token in newTokenStr:
                if token in stopWordList:
                    continue
                else:
                    noStopWordTokenList.append(token)

            for token in newTokenStrEnNum:
                if token in tableNameList or token in programNameList:
                    noStopWordTokenList.append(token)

            newTokenizeStr = " ".join(noStopWordTokenList)
            docList.append(newTokenizeStr)

    return docList, docIndex

# 將既有的SPEC資料集取tfidf，並做PCA壓縮，回傳資料及內有的token與PCA壓縮的model
def all_spec_tfidf_pca(docList):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docList)
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()

    pca = PCA(n_components = 128)
    PCAModel = pca.fit(weight)
    tfPCA = PCAModel.transform(weight)

    return word, tfPCA, PCAModel

# 訓練隨機森林模型
def train_RF_model(tfPCA, docIndex):
    datasetLabel = pd.read_csv("dataset/SPEC_relation_dataset_Agg_df_0.4.csv")
    datasetLabel = datasetLabel.sample(frac=1).reset_index(drop=True)

    dataset = np.zeros(shape=(datasetLabel.shape[0],tfPCA.shape[1]*2))
    labelset = np.zeros(shape=(datasetLabel.shape[0]))

    for i in tqdm(range(datasetLabel.shape[0])):
        spec1 = datasetLabel.iloc[i,:]['SPEC1']
        spec2 = datasetLabel.iloc[i,:]['SPEC2']
        label = datasetLabel.iloc[i,:]['relation']
        spec1_index = docIndex.index(spec1)
        spec2_index = docIndex.index(spec2)
        concat = np.append(tfPCA[spec1_index],tfPCA[spec2_index])
        dataset[i] = concat
        labelset[i] = label

    x_train, x_test, y_train, y_test = train_test_split(dataset, labelset, test_size=0.2, random_state=0)

    print("訓練中請稍後~(若 n_estimators 設越大等越久喔)....")

    forest = ensemble.RandomForestClassifier(n_estimators = 75, n_jobs=6, class_weight="balanced")

    forest_fit = forest.fit(x_train, y_train)
    output = forest_fit.predict(x_test)

    con = pd.crosstab(y_test,output,
            rownames=['label'],
            colnames=['predict'])
    print(con)
    accuracy = accuracy_score(y_test, output)
    precision = precision_score(y_test, output)
    recall = recall_score(y_test, output)
    f1 = f1_score(y_test, output)
    print('accuracy =', accuracy)
    print('precision =', precision)
    print('recall =', recall)
    print('f1 =', f1)

    joblib.dump(forest_fit, 'model/random_forest.pkl')



# 將使用者上傳的資料進行前處理(去除stop word、加入模組與table的專有名詞)
def preprocessing(docx, datasetWords):
  noStopWordTokenList = []
  findEnNum = re.compile("[^A-Za-z0-9]")
  rule = re.compile("[^\u4e00-\u9fa5]")

  docxToText = docx2txt.process(docx)

  docxToTextEnNum = "".join(findEnNum.sub('',docxToText))
  docxToText = "".join(rule.sub('',docxToText))

  tokenizeStr = " ".join(jieba.cut(docxToText)) #未加入stopword
  tokenizeStrEnNum = " ".join(jieba.cut(docxToTextEnNum))

  newTokenStr = tokenizeStr.split(" ")
  newTokenStrEnNum = tokenizeStrEnNum.split(" ")

  for token in newTokenStr:
    if token in stopWordList:
      continue
    else:
      noStopWordTokenList.append(token)

  for token in newTokenStrEnNum:
    if token in tableNameList or token in programNameList:
      noStopWordTokenList.append(token)

  resultList = []
  for token in  noStopWordTokenList:
    if token.lower() in datasetWords:
      resultList.append(token)

  newTokenizeStr = " ".join(resultList)
  print(newTokenizeStr)

  return newTokenizeStr

# 載入RF模型
def load_RF_model(path):
    return joblib.load(path)

# 將使用者上傳的SPEC與資料集中的SPEC一同計算tfidf，並取出屬於使用者上傳的SPEC的vector
def get_tfidf_vec(docxList, TokenizeStr):
  docxList.append(TokenizeStr)
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(docxList)
  word = vectorizer.get_feature_names()
  transformer = TfidfTransformer()
  tfidf = transformer.fit_transform(X)
  weight = tfidf.toarray()
  
  return weight[-1]


# 將使用者上傳的SPEC的vector依照壓縮全部資料集的方式壓縮
def dimendion_reduction(PCAModel, user_vect_weight):
  reshape_weight = user_vect_weight.reshape(1,user_vect_weight.shape[0])
  tfPCA = PCAModel.transform(reshape_weight)
  return tfPCA[0]


# 將壓縮後的vector與資料集的其他資料合併連接
def concat_vec(dataset_vec, user_vec):
  output = []
  for data_vec in dataset_vec:
    output.append(np.append(user_vec,data_vec))

  return np.array(output)


# 預測哪些既有的SPEC是相關的
def RF_predict(pretrainModel, datalist):
  predict = pretrainModel.predict(datalist)
  predict_proba = pretrainModel.predict_proba(datalist)
  return predict, predict_proba[:,1]


# 取出預測結果為相關的部分，並依照由大到小排序
def return_result(result, result_proba, docxNameList):
  output = []
  for i in range(len(docxNameList)):
    if result[i] == 0:
      continue
    else:
      output.append([docxNameList[i], result[i], result_proba[i]])

  output.sort(reverse=True, key = lambda s: s[2])
  return output
