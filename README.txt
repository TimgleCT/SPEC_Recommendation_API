���M�׬� SPEC �����ˡA�i���ѨϥΪ̤W��SPEC���(docx�ɮ�)�A�Өt�αN���˨�L�L�h������SPEC�ɦW�P���������v�A�HJSON�^�Ǳ��˵��G�C�����M�ץثe����ƶ����ȧtAA�z�ߪ��D�{��SPEC�A�Y�n�X�R����˯�O�ݭn�[�s��SPEC�ɮצ�dataset/spec��Ƨ����A�ð���create_dataset.py�H���X�s��SPEC���p��ƶ��C�ӥ��M�׬[�]�� Flask API�A�i���Ѹ󥭥x���A�ȡC���M�׬[�c�P�p��ϥλ����p�U�G

��Ƨ��G
	dataset�G���tSPEC���p���Ҹ�ƶ��ƭ�(csv��)�A��ƶ����ɦW���A�̫᪺�Ʀr�N����s�ɨC�s������cos�ۦ��סA�V���h���p������Y�V�C��spec��Ƨ����t�u���� SPEC docx�ɮסC

	jieba_dict�G���t�����_���ɻݭn���J���r��C

	model�G�x�sNN�PRF�ҫ��C

	new_dataset�G�Y���s��SPEC��ƶ��n�[�J�A�]�ӳz�Lcreate_dataset.py���ͷs����ƶ��ɡA���ͪ��s��ƶ��|�s�b�o�̡C

	pytorch�G���t�w��pytorch���ɮסA�����b�׺ݾ����� pip install (����ɮ׸��|)�A�h pytorch�Y�����w�ˡC


python�ɡG
	api.py�G Flask api �� controller�A�Y�n�Ұʥ�API�h���楻�ɮסC

	create_dataset.py�G�Y�n�[�J�s��SPEC�өݮiSPEC�����˥i�A�Ȫ��d��A�h�bdataset��Ƨ��[�JSPEC��docx�ɫ�A���楻�ɮסA���X���s��ƧY�|�bnew_dataset��Ƨ����C���X����ƧY�ɦW��SPEC_relation_dataset_Agg_df_0.XX�A�̫᪺�Ʀr�N����ƶ����s�ɪ��C�s����cos�ۦ��סC���楻�ɮɡA�i�]�w pca_dimendion(��������)�BcosThreshold(�ҲջP��ƪ�TFIDF�V�q�b�P�w�ۦ��P�_�ɡAcos�ۦ��ת��з�)�BAggThreshold(Agg���s�t��k���Z���֭ȡA�V���h���s����V�e�P�A�V�C�h�V�Y��)�C

	module_common.py�G���Ҳթҧt������k�O�b�ϥ�NN�ҫ��P�H���˪L�ҫ����� SPEC�ɬҷ|�ϥΨ쪺��k�C�ԲӪ���k�����Ьd�\2020 CIP��ߦ��G���i P.19

	module_create_dataset.py�G���Ҳթҧt������k�O�b����create_dataset.py�Х߷s��ƶ��ɷ|�ݭn�ϥΨ쪺��k�C

	module_pytorch_liner.py�G���Ҳթҧt����k���b�ϥ�²��T�hNN���g�����ҫ��ɩҷ|�Ψ쪺��k�C�ԲӪ���k�����Ьd�\2020 CIP��ߦ��G���i P.19 ~ 20

	module_RF.py�G���Ҳթҧt����k���b�ϥ��H���˪L�ҫ��ɩҷ|�Ψ쪺��k�C�ԲӪ���k�����Ьd�\2020 CIP��ߦ��G���i P.20 ~ 21

	pytorch_class.py�G���Ҳդ��w�q�F²��T�hNN�����g�����ҫ����ҫ��[�c�A�䤤�t�� Classifier()���O�C


�n����API�ɩһݪ��ݭn����T�p�U�A�H�U���d��python request�G

	headers = {
    		'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
    		'enctype': "multipart/form-data",
    		'cache-control': "no-cache",
    		'postman-token': "ca4bd255-ddd2-b591-7c28-5deb5eb77828"
    		}


�^�Ǫ�JSON�榡�d�ҡG

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

	
	
	
	
	

