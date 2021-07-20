## Anal Env.
import pandas as pd
import numpy  as np
import os
import re
import datetime
import pytz
import copy
from collections import Counter
from emoji       import UNICODE_EMOJI
from functools   import reduce
import sys
sys.path.append('/home/ez-flow/big_data/python/')
import bigquery_sql_load as sql_loader
import bigquery_etl as bq
import operator
import time
from   datetime import datetime
import pytz

# NLP Env.
import nltk
from nltk import FreqDist
from nltk.corpus   import stopwords
from nltk.tokenize import regexp_tokenize
from nltk.stem     import WordNetLemmatizer,PorterStemmer,LancasterStemmer
from nltk.corpus   import wordnet
from nltk.corpus   import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
import string,re

# stop_words = stopwords.words('english')
import spacy
import gensim
from gensim import corpora
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import ray
import psutil
from tqdm.notebook import tqdm

#glove
from glove import Corpus, Glove

# Vis Env.
import pyLDAvis
import matplotlib.pyplot as plt
import seaborn as sns

# GCP Env.
import google.auth
from google.cloud import bigquery
from googletrans import Translator
from google_trans_new import google_translator

# Coding Env.
import warnings
credentials, project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

client = bigquery.Client(credentials=credentials, project=project_id )
warnings.filterwarnings("ignore")
# ray.shutdown()
#initialization ray
num_logit_cpus = psutil.cpu_count()
ray.init(ignore_reinit_error=True,num_cpus=num_logit_cpus)


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

def lemmatize(x) : 
    if len(x.split(' ')) > 1 : # MWE
        tmp_x = x.split(' ')
        tmp_x = [WordNetLemmatizer().lemmatize(y, pos='v') for y in tmp_x ]
        tokenized_string = " ".join(tmp_x)
    else : # Single
        tokenized_string = WordNetLemmatizer().lemmatize(x, pos='v')
        
    return tokenized_string


def cal_trend_analysis_df(keyword_df,sentiment):
    rem     = string.punctuation
    pattern = r"[{}]".format(rem)

    df = keyword_df
    df['word']   = df['word'].str.replace(pattern, '')
    df['factor_yn'] = ''
    if sentiment == 0:
        factor_col_name = 'cmpl_factor'
        df.loc[((df['word'].isin(cmpl_fc_list))),'factor_yn']    = 'Y'
        df.loc[(~(df['word'].isin(cmpl_fc_list))),'factor_yn']   = 'N'
        df_1         = df.groupby(['yearmonth','year','month','factor_yn']+[factor_col_name])['review_cnt'].sum().reset_index().sort_values(['yearmonth',factor_col_name],ascending=[True,True])
        df_1         = df_1.rename(columns={factor_col_name:f'factor'})
        df_1['sentiment'] = 'negative'
    elif sentiment == 1:
        factor_col_name = 'shiny_factor'
        df.loc[((df['word'].isin(shiny_factor_list))),'factor_yn']    = 'Y'
        df.loc[(~(df['word'].isin(shiny_factor_list))),'factor_yn']   = 'N'
        df_1         = df.groupby(['yearmonth','year','month','factor_yn']+[factor_col_name])['review_cnt'].sum().reset_index().sort_values(['yearmonth',factor_col_name],ascending=[True,True])
        df_1         = df_1.rename(columns={factor_col_name:f'factor'})
        df_1['sentiment'] = 'positive'
    df_1['diff'] = df_1.groupby(['factor'])['review_cnt'].diff().fillna(0)
    window_size  = 2
    cols_name    = 'diff_ma%s'%(window_size)
    df_1[cols_name]   = df_1.groupby('factor')['diff'].transform(lambda x: x.rolling(window_size, 1).mean())
    
    return cols_name,df_1

def make_regidate(regi_df):
    regidate     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    regi_df['regidate'] = regidate
    regi_df['regidate'] = pd.to_datetime(regi_df['regidate'])
    return regi_df


try:
    print(' (Setp 1-1) Load Data')
    ## complain Factor 
    factor_sql    = sql_loader.load_sql(2)
    cpl_factor    = convert_lowercase(bq.select_query(factor_sql))
    cpl_factor    = cpl_factor[~cpl_factor['cmpl_fc1'].isin(['none'])]
    cpl_factor['lemma'] = cpl_factor['synonym'].apply(lambda x : lemmatize(x))
    cpl_factor['porter_stem'] = cpl_factor['synonym'].apply(lambda x : PorterStemmer().stem(x))
    cpl_factor =  cpl_factor.drop(columns=['category']).drop_duplicates()
    cf2_factor = cpl_factor[['cmpl_fc1','cmpl_fc2']].reset_index(drop=True)
    syn_factor = cpl_factor[['cmpl_fc1','synonym']].reset_index(drop=True)
    syn_factor = syn_factor.rename(columns={'synonym':'cmpl_fc2'} )
    lem_factor = cpl_factor[['cmpl_fc1','lemma']].reset_index(drop=True)
    lem_factor = lem_factor.rename(columns={'lemma':'cmpl_fc2'} )
    cpl_factor_merge = pd.concat([cf2_factor,syn_factor,lem_factor]).drop_duplicates()
    cpl_factor_merge = cpl_factor_merge[~cpl_factor_merge['cmpl_fc2'].isin(['none'])]
    cf1_factor = cpl_factor_merge[['cmpl_fc1']].drop_duplicates()
    cmpl_fc_list  = list(set(cpl_factor['cmpl_fc1'].unique().tolist())|set(cpl_factor['cmpl_fc2'].unique().tolist())|set(cpl_factor['synonym'].unique().tolist())\
                    |set(cpl_factor['lemma'].unique().tolist()))
    shiny_factor =  convert_lowercase(bq.select_query(sql_loader.load_sql(6)))
    shiny_factor_list = list(set(shiny_factor['shiny_factor'].unique().tolist())|set(shiny_factor['keyword'].unique().tolist()))


    negative_sql,positive_sql,start_ym,YM = sql_loader.Keyword_Trend_SQL()
    print(f'========================================================================')
    print(f'''Keyword Trend Analysis  :  '{start_ym}'~'{YM}'(target yearmonth) ''')
    print(f'========================================================================')

    print(f' (Step 2-1) Calculate keyword diff and moving average (default 2M)')    
    ## Negative Keyword Pre-processing
    negative_df                           = convert_lowercase(bq.select_query(negative_sql)).sort_values(['word','yearmonth'],ascending=[True,True])
    ## join (Cmpl_fc2 + synonym + lemma)
    neg_keyword_df_join = pd.merge(negative_df,cpl_factor_merge,how='left',left_on='word',right_on=['cmpl_fc2'])
    ## join (Cmpl_fc1)
    neg_keyword_df_join = pd.merge(neg_keyword_df_join,cf1_factor,how='left',left_on='word',right_on=['cmpl_fc1'])
    neg_keyword_df_join.loc[neg_keyword_df_join['cmpl_fc1_x'].isnull(),'cmpl_fc1_x'] = neg_keyword_df_join['cmpl_fc1_y']
    neg_keyword_df_join['cmpl_factor'] = neg_keyword_df_join['cmpl_fc1_x']
    neg_keyword_df_join = neg_keyword_df_join.drop(columns=['cmpl_fc2','cmpl_fc1_x','cmpl_fc1_y'])
    neg_keyword_df_join[~neg_keyword_df_join['cmpl_factor'].isnull()]
    negative_df = neg_keyword_df_join
    negative_df.loc[negative_df['cmpl_factor'].isnull(),'cmpl_factor']   = negative_df['word']
    ## Positive Keyword Pre-processing
    positive_df                           = convert_lowercase(bq.select_query(positive_sql)).sort_values(['word','yearmonth'],ascending=[True,True])
    positive_df.loc[positive_df['shiny_factor'].isnull(),'shiny_factor'] = positive_df['word']
    positive_df.loc[(positive_df['shiny_factor'].isin(['none']))|(positive_df['shiny_factor'].isnull()),'shiny_factor'] = positive_df['word']
    cols_name,neg_trend_df                = cal_trend_analysis_df(negative_df,0)
    cols_name,pos_trend_df                = cal_trend_analysis_df(positive_df,1)
    all_trend_df = pd.concat([neg_trend_df.round(1),pos_trend_df.round(1)])

    # # Make registration date
    all_trend_df['ym_rvw_cnt_rank'] = all_trend_df.groupby(['yearmonth','factor_yn','sentiment'])['review_cnt'].rank(ascending=False,method='first')
    all_trend_df['ym_diff_rank'] = all_trend_df.groupby(['yearmonth','factor_yn','sentiment'])['diff'].rank(ascending=False,method='first')
    all_trend_df = make_regidate(all_trend_df)
    all_trend_tbl_name = 'taxonomy.monthly_all_keyword_trend_anal'
    bq.to_bigquery(all_trend_tbl_name,all_trend_df)
    all_trend_df = all_trend_df.drop(columns=['regidate'])
    print(f'''         >> Success '{all_trend_tbl_name}' DataBase Upload''')


    print(' (Step 3-1) Top30 Keyword Filtering Increased / Decreased by sentiment & factor yn') 
    factor_df_trend              = all_trend_df[(all_trend_df['factor_yn'] =='Y')]
    none_factor_trend            = all_trend_df[(all_trend_df['factor_yn'] =='N')]
    ## top30 increase factor alert list 
    increase_factor_trend                    = factor_df_trend[(factor_df_trend['yearmonth']==YM) & (factor_df_trend['diff'] > 0)].sort_values(['factor_yn','ym_diff_rank'],ascending=[False,True])
    increase_non_factor_trend                = none_factor_trend[(none_factor_trend['yearmonth']==YM) & (none_factor_trend['diff'] > 0) & (none_factor_trend['ym_diff_rank'] <= 30)].sort_values(['factor_yn','ym_diff_rank'],ascending=[False,True])
    increase_factor_df  = pd.concat([increase_factor_trend,increase_non_factor_trend])
    increase_factor_df['trend_cat'] = 'increased'

    ## top30 decrease factor alert List
    decrease_factor_trend                    = factor_df_trend[(factor_df_trend['yearmonth']==YM) & (factor_df_trend['diff'] < 0)].sort_values(['factor_yn','ym_diff_rank'],ascending=[False,False])
    decrease_non_factor_trend                = none_factor_trend[(none_factor_trend['yearmonth']==YM) & (none_factor_trend['diff'] < 0) ].sort_values(['factor_yn','ym_diff_rank'],ascending=[False,False])
    decrease_non_factor_trend                = decrease_non_factor_trend.groupby(['sentiment']).apply(lambda x: x.nlargest(30,['ym_diff_rank'])).reset_index(drop=True)
    decrease_factor_df  = pd.concat([decrease_factor_trend,decrease_non_factor_trend])
    decrease_factor_df['trend_cat'] = 'decreased'

    ## concat increased / decreased trend 
    trend_df_total = pd.concat([increase_factor_df.reset_index(drop=True),decrease_factor_df.reset_index(drop=True)])
    ## Make registration date
    trend_df_total = make_regidate(trend_df_total)
    alert_trend_tbl_name = 'taxonomy.monthly_alert_keyword_trend_anal'
    bq.to_bigquery(alert_trend_tbl_name,trend_df_total)
    print(f'''         >> Success '{alert_trend_tbl_name}' DataBase Upload''')

    print(' (Step 4-1) Top30 reviews count keyword (complain factor / Non-complain Factor)') 
    ##top30 keyword review count by Factor or non_Factor
    factor_top_review_count_word     = factor_df_trend[(factor_df_trend['yearmonth']==YM)].sort_values(['sentiment','ym_rvw_cnt_rank'],ascending=[False,True]).drop(columns=['diff','diff_ma2','ym_diff_rank'])
    nonfactor_top_review_count_word  = none_factor_trend[(none_factor_trend['yearmonth']==YM)].sort_values(['sentiment','ym_rvw_cnt_rank'],ascending=[False,True]).drop(columns=['diff','diff_ma2','ym_diff_rank'])
    nonfactor_top_review_count_word  = nonfactor_top_review_count_word.groupby(['sentiment']).apply(lambda x: x.nsmallest(30,['ym_rvw_cnt_rank'])).reset_index(drop=True)
    top30_word_review_cnt_df         = pd.concat([factor_top_review_count_word.reset_index(drop=True),nonfactor_top_review_count_word.reset_index(drop=True)])
    top30_word_review_cnt_df         = top30_word_review_cnt_df.reindex(columns=['yearmonth', 'year', 'month', 'factor_yn', 'factor', 'review_cnt','ym_rvw_cnt_rank', 'sentiment'])
    ## Make registration date
    top30_word_review_cnt_df = make_regidate(top30_word_review_cnt_df)

    top30_review_cnt_tbl_name = 'taxonomy.monthly_top_review_cnt_keyword_anal'
    bq.to_bigquery(top30_review_cnt_tbl_name,top30_word_review_cnt_df)
    print(f'''         >> Success '{top30_review_cnt_tbl_name}' DataBase Upload''')
except Exception as e:
    print(f'Keyword Trend Error : {e}')
