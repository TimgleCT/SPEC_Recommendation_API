from flask import Flask, jsonify, request
import numpy as np
import os
import module_pytorch_liner
import module_common
import module_RF

app = Flask(__name__)
app.config["DEBUG"] = True

# for pytorch liner 模型的部分
labelSpecList = module_common.get_label_file()
dfTFIDF, kw_string_list, dataset_key_word = module_pytorch_liner.tfidf(labelSpecList)
dfTFIDF_PCA, tfidfPCA, PCAReductor = module_pytorch_liner.PCA_dimension_reduction(dfTFIDF, labelSpecList)



# for sklearn RF 模型的部分
docxList, docIndex = module_RF.preprocessing_allSPEC('dataset/spec/')
docxWord, allSPECPCA, PCAProcesser = module_RF.all_spec_tfidf_pca(docxList)
module_RF.train_RF_model(allSPECPCA, docIndex)



# 測試API能否連接
@app.route('/api/content', methods=['POST'])
def content_api_test():
    data = request.get_json()
    print(data)
    myString = data["String"]
    output = module_common.testAPI(myString)
    return jsonify(output)




# 使用者使用本服務所需訪問之路徑－使用pytroch訓練出來的liner model
@app.route('/api/get_related_files_liner', methods=['POST'])
def get_related_files_liner():

    # 接收到使用者上傳spec文件
    uploadFile = module_common.load_docxfile(request)

    # 將使用者上傳的spec文件與資料集中的其他文件一起計算tfidf
    user_tfidf_vect = module_pytorch_liner.tfidf_user_upload(uploadFile, kw_string_list, dataset_key_word)

    # 將使用者上傳的spec的tfidf向量丟進之前壓縮所有文件時的PCA進行壓縮，產出128維的向量
    user_tfidf_pca = module_pytorch_liner.PCA_dimension_reduction_user_upload(user_tfidf_vect, PCAReductor)

    # 使用者spec的tfidf向量被PCA壓縮後，與其他文件被壓縮的向量合併成256維，所以輸出為(文件數*256)的陣列
    preprocess_data = module_pytorch_liner.concat_user_dataset_V(user_tfidf_pca, tfidfPCA)

    # 將剛剛所得之(文件數*256)陣列一筆一筆的去預測，並得出預測結果
    predict_result_list = module_pytorch_liner.predict_loop(preprocess_data, labelSpecList)

    # 將預測結果依照相關性的機率排序
    predict_list_sorted = module_pytorch_liner.sort_predicted_list(predict_result_list)

    # 取出預測相關的文章，若相關性機率>0.9的文章超過30個，則只印出相關機率大於0.9的文章
    top_related_doc = module_common.get_top_related_doc(predict_list_sorted)

    # 若無相關SPEC可推薦則回傳
    if len(top_related_doc) < 1:
        return "無相關SPEC可供推薦"

    # 將結果轉成json格式
    related_doc_json = module_common.result_list_to_json(top_related_doc)

    for one_doc in top_related_doc:
        print(one_doc)

    return related_doc_json





# 使用者使用本服務所需訪問之路徑－使用sklearn訓練出的隨機森林模型
@app.route('/api/get_related_files_RF', methods=['POST'])
def get_related_files_RF():

    # 接收到使用者上傳spec文件
    uploadFile = module_common.load_docxfile(request)

    # 載入RF模型
    RFModel = module_RF.load_RF_model('model/random_forest.pkl')

    # 將使用者上傳的資料前處理(斷詞、去除stopword、去除符號與換行)
    userSPEC = module_RF.preprocessing(uploadFile, docxWord)

    # 將使用者上傳的spec文件與資料集中的其他文件一起計算tfidf
    userSPECWeight = module_RF.get_tfidf_vec(docxList, userSPEC)

    # 將使用者上傳的spec的tfidf向量丟進之前壓縮所有文件時的PCA進行壓縮，產出128維的向量
    user_pca_vect = module_RF.dimendion_reduction(PCAProcesser, userSPECWeight)

    # 使用者spec的tfidf向量被PCA壓縮後，與其他文件被壓縮的向量合併成256維，所以輸出為(文件數*256)的陣列
    datalist = module_RF.concat_vec(allSPECPCA, user_pca_vect)

    # 將整個陣列放入預測，產出相關結果與機率
    result, result_proba = module_RF.RF_predict(RFModel, datalist)

    # 將預測結果依照相關性的機率排序
    resultList = module_RF.return_result(result, result_proba, docIndex)

    # 取出預測相關的文章，若相關性機率>0.9的文章超過30個，則只印出相關機率大於0.9的文章
    top_related_doc = module_common.get_top_related_doc(resultList)

    # 若無相關SPEC可推薦則回傳
    if len(top_related_doc) < 1:
        return "無相關SPEC可供推薦"

    # 將結果轉成json格式
    related_doc_json = module_common.result_list_to_json(top_related_doc)

    for one_doc in top_related_doc:
        print(one_doc)

    return related_doc_json


if __name__ == '__main__':
    app.run()