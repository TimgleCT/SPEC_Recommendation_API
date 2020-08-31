本專案為 SPEC 文件推薦，可提供使用者上傳SPEC文件(docx檔案)，而系統將推薦其他過去相關的SPEC檔名與相關的機率，以JSON回傳推薦結果。但本專案目前的資料集內僅含AA理賠的主程式SPEC，若要擴充其推薦能力需要加新的SPEC檔案至dataset/spec資料夾內，並執行create_dataset.py以產出新的SPEC關聯資料集。而本專案架設成 Flask API，可提供跨平台的服務。本專案架構與如何使用說明如下：

資料夾：
	dataset：內含SPEC關聯標籤資料集數個(csv檔)，資料集的檔名中，最後的數字代表分群時每群的平均cos相似度，越高則關聯條件更嚴苛。而spec資料夾內含真正的 SPEC docx檔案。

	jieba_dict：內含結疤斷詞時需要載入的字典。

	model：儲存NN與RF模型。

	new_dataset：若有新的SPEC資料集要加入，因而透過create_dataset.py產生新的資料集時，產生的新資料集會存在這裡。

	pytorch：內含安裝pytorch的檔案，直接在終端機執行 pip install (兩個檔案路徑)，則 pytorch即完成安裝。


python檔：
	api.py： Flask api 的 controller，若要啟動本API則執行本檔案。

	create_dataset.py：若要加入新的SPEC來拓展SPEC文件推薦可服務的範圍，則在dataset資料夾加入SPEC的docx檔後，執行本檔案，產出的新資料即會在new_dataset資料夾內。產出的資料即檔名為SPEC_relation_dataset_Agg_df_0.XX，最後的數字代表本資料集分群時的每群平均cos相似度。執行本檔時，可設定 pca_dimendion(降維維度)、cosThreshold(模組與資料表TFIDF向量在判定相似與否時，cos相似度的標準)、AggThreshold(Agg分群演算法的距離閥值，越高則分群條件越寬鬆，越低則越嚴謹)。

	module_common.py：此模組所含有的方法是在使用NN模型與隨機森林模型推薦 SPEC時皆會使用到的方法。詳細的方法說明請查閱2020 CIP實習成果報告 P.19

	module_create_dataset.py：此模組所含有的方法是在執行create_dataset.py創立新資料集時會需要使用到的方法。

	module_pytorch_liner.py：此模組所含之方法為在使用簡單三層NN神經網路模型時所會用到的方法。詳細的方法說明請查閱2020 CIP實習成果報告 P.19 ~ 20

	module_RF.py：此模組所含之方法為在使用隨機森林模型時所會用到的方法。詳細的方法說明請查閱2020 CIP實習成果報告 P.20 ~ 21

	pytorch_class.py：此模組內定義了簡單三層NN類神經網路模型的模型架構，其中含有 Classifier()類別。


要接本API時所需的需要的資訊如下，以下為範例python request：

	headers = {
    		'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
    		'enctype': "multipart/form-data",
    		'cache-control': "no-cache",
    		'postman-token': "ca4bd255-ddd2-b591-7c28-5deb5eb77828"
    		}


回傳的JSON格式範例：

[{"docName":"UCAAA0_0700.docx","relation":"1.0","probability":"0.8533333333333334"},
{"docName":"UCAAB1_B006.docx","relation":"1.0","probability":"0.8533333333333334"},
{"docName":"UCAAJ0_0600.docx","relation":"1.0","probability":"0.8533333333333334"},
{"docName":"UCAAH3_B401.docx","relation":"1.0","probability":"0.8266666666666667"},
{"docName":"UCAAZZ_B003.docx","relation":"1.0","probability":"0.8"},
{"docName":"UCAAJ0_0100.docx","relation":"1.0","probability":"0.7866666666666666"},
{"docName":"UCAAJ1_0110.docx","relation":"1.0","probability":"0.7866666666666666"},
{"docName":"UCAAP0_0100.docx","relation":"1.0","probability":"0.7866666666666666"},
{"docName":"UCAAB1_B005.docx","relation":"1.0","probability":"0.7466666666666667"},
{"docName":"UCAAC0_0800.docx","relation":"1.0","probability":"0.72"},
{"docName":"UCAAD0_0900.docx","relation":"1.0","probability":"0.7066666666666667"},
{"docName":"UCAAQ0_B002.docx","relation":"1.0","probability":"0.7066666666666667"},
{"docName":"UCAAC0_0801.docx","relation":"1.0","probability":"0.6933333333333334"},
{"docName":"UCAAB0_0100.docx","relation":"1.0","probability":"0.68"},
{"docName":"UCAAJ1_B002.docx","relation":"1.0","probability":"0.68"},
{"docName":"UCAAA0_0102.docx","relation":"1.0","probability":"0.6666666666666666"},
{"docName":"UCAAB1_B009.docx","relation":"1.0","probability":"0.6266666666666667"},
{"docName":"UCAAM2_B403.docx","relation":"1.0","probability":"0.6"},
{"docName":"UCAAQ0_B001.docx","relation":"1.0","probability":"0.5866666666666667"},
{"docName":"UCAAJ1_B001.docx","relation":"1.0","probability":"0.5333333333333333"}]

	
	
	
	
	

