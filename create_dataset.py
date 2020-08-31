import os
import docx
import docx2txt
import re
import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

import module_create_dataset

pca_dimendion = 256
cosThreshold = 0.5
AggThreshold = 2.0

docDictList = module_create_dataset.getTableModel()
TFIDF_T, TFIDF_M, fileIndex = module_create_dataset.getTableModelTFIDF(docDictList)

tableRelationMatrix = module_create_dataset.getRelationMatrix(cosThreshold, TFIDF_T, fileIndex, "new_dataset/relateMatrix_df_table.csv")
modelRelationMatrix = module_create_dataset.getRelationMatrix(cosThreshold, TFIDF_M, fileIndex, "new_dataset/relateMatrix_df_model.csv" )

docxList = module_create_dataset.preprocessSPEC("dataset/spec/")
docxWeight = module_create_dataset.docxTFIDF(docxList)

# targetThreshold = module_create_dataset.findBestThreshold(23, 25, 0.35)
aggClusterResult, cosScore = module_create_dataset.AgglomerativeCluster(docxWeight, pca_dimendion, AggThreshold)

clusterRelateMatrix = module_create_dataset.getRelateMatrixFromCluster(aggClusterResult, fileIndex, "new_dataset/clusterRelateAgg"+ str(AggThreshold) +"_df.csv")

finalRelateMatrix = module_create_dataset.outputFinalRelateMatrix(clusterRelateMatrix, modelRelationMatrix, tableRelationMatrix, fileIndex)

module_create_dataset.createDataset(finalRelateMatrix, fileIndex, "new_dataset/SPEC_relation_dataset_Agg_df_"+ str(cosScore) +".csv")

