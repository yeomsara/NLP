import sys
sys.path.append('/home/ez-flow/big_data/python/')
import pandas as pd
import numpy  as np
import os
import re
import datetime
import pytz
import copy
from collections import Counter
from functools   import reduce
import bigquery_sql_load as sql_loader
import bigquery_etl as bq
import operator
import time
from IPython.display import display
from   datetime import datetime
import pytz
# stop_words = stopwords.words('english')
import spacy
import gensim
from gensim import corpora
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import ray
import psutil
from tqdm.notebook import tqdm

# GCP Env.
import google.auth
from google.cloud import bigquery
from googletrans import Translator
from google_trans_new import google_translator

# Coding Env.
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import ast

credentials, project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

client = bigquery.Client(credentials=credentials, project=project_id )
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#initialization ray
num_logit_cpus = psutil.cpu_count()

# Load Data FROM Big Query(db connection)
def convert_lowercase(df):
    df_1 =  df.apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)
    upper_list = ['reviewId','asin','size','cmpl_fc1_cd']
    cols = list(set(upper_list)& set(df_1.columns))
    df_1[cols] = df_1[cols].apply(lambda x: x.astype(str).str.upper() if(x.dtype == 'object') else x)
    return df_1

def convert_uppercase(df):
    upper_list = ['reviewId','asin']
    cols = list(set(upper_list)& set(df.columns))
    df[cols] = df[cols].apply(lambda x: x.astype(str).str.upper() if(x.dtype == 'object') else x)
    return df

def make_regidate(regi_df):
    timezone     = pytz.timezone('Asia/Seoul')
    regidate     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    regi_df['regidate'] = regidate
    regi_df['regidate'] = pd.to_datetime(regi_df['regidate'])
    return regi_df

##### Pipeline Start ######
try:
    bert_result_sql,taxonomy_result_sql,start_ym,end_ym = sql_loader.Hybrid_Model_input()
    print(f'''==================================================================================''')
    print(f''' Hybrid Model date yearmonth     between '{start_ym}' and '{end_ym}' ''')      
    print(f'''==================================================================================''')
    print(' (Setp 1-1) Load Data')
    bert_df     = convert_lowercase(bq.select_query(bert_result_sql))
    taxonomy_df = convert_lowercase(bq.select_query(taxonomy_result_sql))
    asin_cat_df = convert_lowercase(bq.select_query(sql_loader.monthly_asin_cat_master()))
    asin_cat_df = asin_cat_df.drop_duplicates(['reviewId','asin'])
    print(f'''          date yearmonth     between '{start_ym}' and '{end_ym}' Bert Predict DF     : {bert_df.shape} ''')
    print(f'''          date yearmonth     between '{start_ym}' and '{end_ym}' Taxonomy Predict DF : {taxonomy_df.shape}''')
    print(f'''          date yearmonth     between '{start_ym}' and '{end_ym}' asin category master DF : {asin_cat_df.shape}''')
    label_cmpl_cf_dict = {  'recovery'         : 'CF011', 
                            'durability'       : 'CF001',
                            'defective'        : 'CF002', 
                            'too soft'         : 'CF013', 
                            'too hard'         : 'CF012',
                            'missing parts'    : 'CF003',
                            'odor'             : 'CF017',
                            'sound'            : 'CF004',
                            'uncomfortable'    : 'CF014',
                            'size issue'       : 'CF008',
                            'shipping damage'  : 'CF018',
                            'delivery'         : 'CF019',
                            'fiberglass'       : 'CF007',
                            'hard to set up'   : 'CF010', 
                            'slipping'         : 'CF009', 
                            'cover issue'      : 'CF006', 
                            'customer service' : 'CF021', 
                            'springs felt'     : 'CF016',
                            'overall quality'  : 'CF005',
                            'no support'       : 'CF015',
                            'customer error'   : 'CF020',
                            'others'           : 'CF999', 
                            'structure design' : 'CF022'
                       }

    label_dict_inverse = {v: k for k, v in label_cmpl_cf_dict.items()}
    print(' (Setp 2-1) Bert predict result preprocessing ')
    bert_result = bert_df.rename(columns=label_dict_inverse)
    bert_result['y_pred_list'] =bert_result['y_pred_list'].apply(lambda x : ast.literal_eval(x)) 
    bert_result['y_pred_prob'] =bert_result['y_pred_prob'].apply(lambda x : ast.literal_eval(x)) 

    bert_predct_df = pd.DataFrame()
    for i in range(0,bert_result.shape[0]):
        rev =  bert_result.loc[i,'reviewId']
        b_asin =  bert_result.loc[i,'asin']
        yearm =  bert_result.loc[i,'yearmonth']
        tt2 = pd.DataFrame(list(zip(bert_result.loc[i,'y_pred_list'],bert_result.loc[i,'y_pred_prob'])))
        tt2.columns = ['cmpl_fc1','bert_prob']
        tt2['reviewId']  = rev
        tt2['asin']      = b_asin
        tt2['yearmonth'] = yearm
        bert_predct_df = pd.concat([bert_predct_df,tt2])

    bert_predict_df = bert_predct_df.reindex(columns=['reviewId','yearmonth','asin','cmpl_fc1','bert_prob'])
    print(' (Setp 3-1) Taxonomy result preprocessing ')
    taxonomy_predict_df = taxonomy_df[['reviewId','yearmonth','asin','cmpl_fc1','softmax_similarity']] 
    taxonomy_predict_df.columns = ['reviewId','yearmonth','asin','cmpl_fc1','taxonomy_similarity']
    print(' (Setp 4-1) Merge taxonomy + Bert predict result ')
    hybrid_model = pd.merge(taxonomy_predict_df,bert_predict_df,how='outer',on=['reviewId','yearmonth','asin','cmpl_fc1'])
    hybrid_model = hybrid_model.fillna(0)
    hybrid_model['similarity_sum'] = hybrid_model['taxonomy_similarity']+hybrid_model['bert_prob'] 
    hybrid_model['exp_similarity'] = hybrid_model.groupby('reviewId')['similarity_sum'].apply(np.exp)
    hybrid_model['SUM_EXP_PROB']   = hybrid_model.groupby('reviewId')['exp_similarity'].transform('sum')
    hybrid_model['similarity']      = hybrid_model['exp_similarity'] / hybrid_model['SUM_EXP_PROB']
    hybrid_result = hybrid_model.sort_values(['yearmonth','reviewId','similarity'],ascending=[False,True,False])
    hybrid_result['cumsum_similarity'] = hybrid_result.groupby(['reviewId'])['similarity'].cumsum()
    hybrid_result = hybrid_result.loc[(hybrid_result['similarity'] >= 0.3)]
    hybrid_result['similarity_rank']   = hybrid_result.groupby(['reviewId'])['similarity'].rank(ascending=False)
    hybrid_result = hybrid_result.drop(columns=['exp_similarity','SUM_EXP_PROB'])
    hybrid_result = pd.merge(hybrid_result,asin_cat_df[['reviewId','asin','review_text']],how='left',on=['reviewId','asin'])
    hybrid_result = hybrid_result.reindex(columns=['reviewId', 'asin','yearmonth','cmpl_fc1', 'taxonomy_similarity', 'bert_prob','similarity_sum', 'similarity', 
                                                   'cumsum_similarity', 'similarity_rank', 'review_text'])
    
    
    reviewId_list = "','".join(hybrid_result.reviewId.unique())
    hybrid_result = make_regidate(hybrid_result)
    ##Hybrid result db upload
    print(' (Step 5-1) connection DataBase ')
    ## avoid data duplicates upload delete reviewId and upload
    hybrid_tbl_name = 'taxonomy.hybrid_model_cf1_result'
    bq.excute_query(f''' DELETE FROM {hybrid_tbl_name} WHERE reviewId in({"'"+reviewId_list+"'"}) ''')
    print(f''' (Step 5-2) delete '{hybrid_tbl_name}' target yearmonth between '{start_ym}' and '{end_ym}' reviewId count {len(hybrid_result.reviewId.unique())}''')
    bq.insert_append_query(hybrid_tbl_name,hybrid_result)
    print(f'           >> Success Hybrid Pipeline')
    print(f'           >> Success {hybrid_tbl_name} DataBase Upload')
except Exception as e:
    print(f'Hybrid Model Error : {e}')
