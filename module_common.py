import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename

# 測試API是否能運作
def testAPI(myString):
    return "你傳過來的字串為：" + myString

# 允許上傳的檔案類型
def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['docx'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 取的使用者上傳的docx檔
def load_docxfile(request):
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print("上傳"+str(filename)+"檔案成功!")
        return file
    return "上傳檔案未成功!"

# 載入已label的資料檔名
def get_label_file():
    label_file = pd.read_csv('dataset/SPEC_relation_dataset_Agg_df_0.3.csv')
    return list(set(list(label_file["SPEC1"]) + list(label_file["SPEC2"])))

# 取出預測是相關文章且若相關機率大於0.9的文章數超過30個的話，則回傳相關機率>0.9的全部資料，不然回傳前20個相關文件
def get_top_related_doc(sort_predicted_list):
    flag = 1
    flag_index = 0
    index = 0
    without_unrelate_doc = []
    for element in sort_predicted_list:
        if element[1] == 0:
            break
        else:

            if element[2] > 0.9 and element[2] < flag:
                flag = sort_predicted_list[index][2]
                flag_index = index

            without_unrelate_doc.append(element)
        
        index += 1
    
    if flag_index > 30:
        print("關聯機率高於0.9的超過30個!")
        return without_unrelate_doc[:flag_index]

    return without_unrelate_doc

# 將結果陣列轉成json
def result_list_to_json(related_doc_list):
    related_doc_array = np.array(related_doc_list)
    df = pd.DataFrame(related_doc_array, columns = ['docName','relation','probability'])
    json = df.to_json(orient='records')
    return json