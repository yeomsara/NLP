## Anal Env.
import pandas as pd
import numpy  as np
import os
import string,re
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
print(f'multiprocessing using {num_logit_cpus} cores')
ray.init(ignore_reinit_error=True,num_cpus=num_logit_cpus)


# Define List

# Load Data "FROM" Big Query(db connection)
def convert_lowercase(df):
    df_1 =  df.apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)
    return df_1

def convert_uppercase(df):
    upper_list = ['reviewId','asin','sku','size','cmpl_fc1_cd']
    cols = list(set(upper_list)& set(df.columns))
    df[cols] = df[cols].apply(lambda x: x.astype(str).str.upper() if(x.dtype == 'object') else x)
    return df

def top20_df_brand(df):
    br_cat_rvw_rank  = pd.pivot_table(df, index = ['brand'], values = ['reviewId'], columns = ['prdct_ctgry_4_5'], aggfunc = ['count'], fill_value = 0, margins = True)#.reset_index()#.to_csv('ddd.csv')
    br_rvw_rank_all  = br_cat_rvw_rank['count']['reviewId']['All'].reset_index()
    br_rvw_rank_all  = br_rvw_rank_all.loc[(br_rvw_rank_all['brand'] != 'All'),]
    br_rvw_rank_all['rank'] = br_rvw_rank_all['All'].rank(ascending=False).astype(int)
    br_rvw_rank_all  = br_rvw_rank_all.sort_values(by='rank',ascending=True)
    br_rvw_rank_all  = br_rvw_rank_all[0:20]
    return br_rvw_rank_all['brand'].tolist()

# Lemmatize
def lemmatize(x) : 
    if len(x.split(' ')) > 1 : # MWE
        tmp_x = x.split(' ')
        tmp_x = [WordNetLemmatizer().lemmatize(y, pos='v') for y in tmp_x ]
        tokenized_string = " ".join(tmp_x)
    else : # Single
        tokenized_string = WordNetLemmatizer().lemmatize(x, pos='v')
        
    return tokenized_string

# (Step1-1) data filtering  
def make_anal_df(df,senti):
    ## sentiment (0) : negative review | (1) : positive review
    df['rat_sentiment'] =  np.where(df['rating']<=2, 0,1) ## give rating sentiment 1~2 star = neg /  5 star = pos
    ## combind title + review_text
    df['review_text']   = df[['title','review_text']].astype(str).sum(axis=1)
    if senti == 0:
        df_1 = df[(df['rat_sentiment']==0)]
    else :
        df_1 = df[(df['rat_sentiment']==1)]    
    df_1['date'] = pd.to_datetime(df_1.date)
    df_1['yearmonth'] = df_1['date'].dt.strftime('%Y%m')
    df_1['year']      = df_1['date'].dt.strftime('%Y')
    df_1['month']     = df_1['date'].dt.strftime('%m')
    df_1 = convert_uppercase(df_1)
    return df_1


#check multiprocessing progress 
def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])
    
def ray_multiprocessing_progress(ray_df):
    for x in tqdm(to_iterator(ray_df), total=len(ray_df)):
        pass
    ray_df  = pd.concat(ray.get(ray_df))
    return ray_df

def make_regidate(regi_df):
    regidate     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    regi_df['regidate'] = regidate
    regi_df['regidate'] = pd.to_datetime(regi_df['regidate'])
    return regi_df

@ray.remote
def reviewId_tokenized_sents(df):
    review_df   = df
    N_POS_TAG   = ['CC','CD','DT','EX','FW','LS','PDT','POS','PRP','PRP$','TO','WDT','WP','WRB']
    Y_POS_TAG   = ['JJ','JJR','JJS','MD','IN','NN','NNS','NNP','NNPS','RB','RBR','RBS','RP','UH','VB','VBG','VBD','VBN','VBP','VBZ']
    keyword_df2 = pd.DataFrame()
    mwe_tokenizer          = nltk.tokenize.MWETokenizer(mwe,separator=' ')
    for i,v in enumerate(review_df['review_text']):
        corpus = []
        reid = str(review_df.iloc[i].reviewId)
        asin = str(review_df.iloc[i].asin)
        ym   = str(review_df.iloc[i].yearmonth)
        for j in pos_tag(regexp_tokenize(v,"[\w']+")) :
            if (j[1] in Y_POS_TAG ) & (len(j[0])>1) & (j[0].isascii()):
                    corpus.append(j[0])
            tokenized_string       = [WordNetLemmatizer().lemmatize(x,pos='v') for x in corpus]
            tokenized_string       = mwe_tokenizer.tokenize(tokenized_string)
            tokenized_string       = [i for i in tokenized_string if i not in stop_words]
            word_df    = pd.DataFrame(tokenized_string,columns=['word'])
            keyword_df = pd.DataFrame.from_dict(Counter(word_df['word']), orient='index').reset_index()
            keyword_df['reviewId'] = reid
            keyword_df['asin'] = asin
            keyword_df['yearmonth'] = ym
            if keyword_df.shape[0] > 0:
                keyword_df.columns = ['word','count','reviewId','asin','yearmonth']    
        keyword_df2    = pd.concat([keyword_df2,keyword_df])
        keyword_df2    = keyword_df2[~keyword_df2['word'].isin(stop_words)].sort_values(by='reviewId',ascending=False)
        
    return keyword_df2

##### Pipeline Start ######
try:
    ##  Data Load
    review_sql,start_date,end_date,start_ym,end_ym   = sql_loader.review_input_load_sql()
    ##########################################
    # sql_cd 1 == stopword sql     
    # sql_cd 2 == complain Factor sql     
    # sql_cd 3 == bsr_brnad sql     
    # sql_cd 4 == part_sql sql    
    # sql_cd 5 == taxonomy rule sql 
    # sql_cd 6 == shiny factor 
    ##########################################
    filter_sql    = sql_loader.load_sql(1)
    factor_sql    = sql_loader.load_sql(2)
    bsr_brand_sql = sql_loader.load_sql(3)
    part_sql      = sql_loader.load_sql(4)
    shiny_sql     = sql_loader.load_sql(6)
    print(f'''==================================================================================''')
    print(f''' keyword Frequency target initialTime between '{start_ym}' and '{end_ym}' ★{end_date}(batch time)''')
    print(f''' keyword Frequency date yearmonth     between '{start_ym}' and '{end_ym}' ''')      
    print(f'''==================================================================================''')
    print(' (Setp 1-1) Load data')
    ##(P2) Load complain factor dataframe & multi word express
    cpl_factor    = convert_lowercase(bq.select_query(factor_sql))
    cpl_factor['lemma'] = cpl_factor['synonym'].apply(lambda x : lemmatize(x))
    cpl_factor['porter_stem'] = cpl_factor['synonym'].apply(lambda x : PorterStemmer().stem(x))
    shiny_factor = convert_lowercase(bq.select_query(shiny_sql))
    shiny_factor['lemma'] = shiny_factor['keyword'].apply(lambda x : lemmatize(x))
    shiny_factor.columns = ['shiny_factor','word','lemma']

    factor_list  = list(set(cpl_factor['cmpl_fc1'].unique().tolist())|set(cpl_factor['cmpl_fc2'].unique().tolist())|set(cpl_factor['synonym'].unique().tolist())\
                         |set(cpl_factor['lemma'].unique().tolist())|set(shiny_factor['word'].unique().tolist())|set(shiny_factor['lemma'].unique().tolist()))
    multi_express = list(filter(lambda x: len(x.split(' '))>1 , factor_list))
    ## Put this list for MWE Tokenizing
    mwe = [tuple( f.split(' ')) for f in multi_express]

    ##(P3) Load Review & BSR & Parts & stopwords & taxonomy rule dataframe
    bsr_brand_df   = convert_lowercase(bq.select_query(bsr_brand_sql))
    top_brand      = list(set(bsr_brand_df['brand'].unique()))
    df             = convert_lowercase(bq.select_query(review_sql))
    stopword_df    = convert_lowercase(bq.select_query(filter_sql))
    parts_df       = convert_lowercase(bq.select_query(part_sql))
    parts_df['part_lemma'] = parts_df['part_word'].apply(lambda x : WordNetLemmatizer().lemmatize(x,pos='v'))
    parts_list     = list(set(parts_df['part_word'].unique().tolist())|set(parts_df['part_lemma'].unique().tolist()))
    parts_list.sort(reverse=False)
    stop_words     = list(set(stopword_df['stopword'])|set(parts_list))
    df             = convert_lowercase(bq.select_query(review_sql))
    df['rat_sentiment'] =  np.where(df['rating']<=2, 0,1) ## give rating sentiment 1~2 star = neg /  5 star = pos
    print(f'''           yearmonth between '{start_ym}' and '{end_ym}' review data : ''',df.shape)
    print(f'''           yearmonth between '{start_ym}' and '{end_ym}' Positive(rating 5 star) data : ''',df[df['rating'].isin([5])].shape)
    print(f'''           yearmonth between '{start_ym}' and '{end_ym}' Negative(rating 1~2 star) data : ''',df[df['rating'].isin([1,2])].shape)
    neg_df = make_anal_df(df,0)
    pos_df = make_anal_df(df,1)
    all_df = pd.concat([neg_df,pos_df])
    print(f'           rating(1~2) negative review : {neg_df.shape}')
    print(f'           rating(5) positive review   : {pos_df.shape}')
    print(f'           all reviews : {all_df.shape}')
    print(f'           yeamonth review_data : \n' ,all_df.groupby(['yearmonth','rat_sentiment'])['reviewId'].count().reset_index())

    print(' (Setp 2-1) Tokenized & Make Corpus')
    all_keyword_df = [reviewId_tokenized_sents.remote(all_df.loc[(all_df['yearmonth']==j)]) for j in tqdm(all_df['yearmonth'].unique()) ]
    all_keyword_df = pd.concat(ray.get(all_keyword_df))
#     all_keyword_df = ray_multiprocessing_progress(all_keyword_df)
    all_keyword_df = convert_uppercase(all_keyword_df.drop(columns='index').reset_index(drop=True))


    key_cols = ['reviewId', 'yearmonth','year', 'month', 'rating']
    merge_all_df = convert_uppercase(all_df[key_cols])
    all_keyword = pd.merge(all_keyword_df,merge_all_df,how='left',on=['reviewId','yearmonth'])

    rem = string.punctuation
    pattern = r"[{}]".format(rem)
    all_keyword['word'] = all_keyword['word'].str.replace(pattern, '')
    
    print(' (Setp 2-2)  Divided positive & negative')
    pos_all_keyword = all_keyword[all_keyword['rating'].isin([5])]
    neg_all_keyword = all_keyword[all_keyword['rating'].isin([1,2])]
    print(f'           rating(1~2) negative keyword : {neg_all_keyword.shape}')
    print(f'           rating(5) positive keyword   : {pos_all_keyword.shape}')
    print(f'           neg+pos review keyword       : {all_keyword.shape}')

    pos_all_keyword['sentiment'] = 'positive'
    neg_all_keyword['sentiment'] = 'negative'

    print(' (Setp 3-1)  Join Factor')
    cmpl_fc_list2  = pd.DataFrame(list(set(cpl_factor['cmpl_fc1'].unique().tolist())|set(cpl_factor['cmpl_fc2'].unique().tolist())|set(cpl_factor['synonym'].unique().tolist())|set(cpl_factor['lemma'].unique().tolist())),columns=['cmpl_factor'])
#     shiny_factor2  = pd.DataFrame(list(set(shiny_factor['shiny_factor'].unique().tolist())|set(shiny_factor['word'].unique().tolist())|set(shiny_factor['lemma'].unique().tolist())),columns=['shiny_factor'])
    neg_keyword = pd.merge(neg_all_keyword,cmpl_fc_list2,how='left',left_on=['word'],right_on=['cmpl_factor']).fillna('')
    pos_keyword =  pd.merge(pos_all_keyword,shiny_factor[['word','shiny_factor']],how='left',on=['word']).fillna('')

    neg_group_cols = ['asin','word','yearmonth','year','month','rating','sentiment','cmpl_factor']
    pos_group_cols = ['asin','word','yearmonth','year','month','rating','sentiment','shiny_factor']

    print(' (Setp 4-1)  Calculate count of reviews ')
    neg_keyword_freq       = neg_keyword.groupby(neg_group_cols)['count'].sum().reset_index()
    neg_keyword_review_cnt = neg_keyword.groupby(neg_group_cols)['reviewId'].count().reset_index()
    pos_keyword_freq       = pos_keyword.groupby(pos_group_cols)['count'].sum().reset_index()
    pos_keyword_review_cnt = pos_keyword.groupby(pos_group_cols)['reviewId'].count().reset_index()

    neg_keyword_table = pd.merge(neg_keyword_freq,neg_keyword_review_cnt,how='left',on=neg_group_cols)
    pos_keyword_table = pd.merge(pos_keyword_freq,pos_keyword_review_cnt,how='left',on=pos_group_cols)

    neg_keyword_table = neg_keyword_table.rename(columns ={'count':'word_cnt','reviewId':'review_cnt'})
    pos_keyword_table = pos_keyword_table.rename(columns ={'count':'word_cnt','reviewId':'review_cnt'})
    
    
    print(' (Step 5-1) connection DataBase ')
    neg_db_cols = ['asin','yearmonth','year','month','rating','word','cmpl_factor','word_cnt','review_cnt','sentiment']
    pos_db_cols = ['asin','yearmonth','year','month','rating','word','shiny_factor','word_cnt','review_cnt','sentiment']
    neg_keyword_table = neg_keyword_table.reindex(columns=neg_db_cols)
    pos_keyword_table = pos_keyword_table.reindex(columns=pos_db_cols)
    
    neg_keyword_table =make_regidate(neg_keyword_table) 
    pos_keyword_table =make_regidate(pos_keyword_table)
    
    neg_keyword_tbl_name = 'taxonomy.negative_keyword_anal'
    pos_keyword_tbl_name = 'taxonomy.positive_keyword_anal'

    bq.excute_query(f''' DELETE FROM {neg_keyword_tbl_name} WHERE yearmonth between '{start_ym}' and '{end_ym}' ''')
    print(f''' (Step 5-2) delete '{neg_keyword_tbl_name}' table target yearmonth between '{start_ym}' and '{end_ym}' ''')
    bq.excute_query(f''' DELETE FROM {pos_keyword_tbl_name} WHERE yearmonth between '{start_ym}' and '{end_ym}' ''')
    print(f''' (Step 5-2) delete '{pos_keyword_tbl_name}' table target yearmonth between '{start_ym}' and '{end_ym}' ''')

    bq.insert_append_query(neg_keyword_tbl_name,neg_keyword_table)
    print(f'Success {neg_keyword_tbl_name} DataBase Upload')
    bq.insert_append_query(pos_keyword_tbl_name,pos_keyword_table)
    print(f'Success {pos_keyword_tbl_name} DataBase Upload')
    print(f'Success Keyword anal Predict Pipeline')
except Exception as e :
    print(f'Keyword Analysis Error : {e}')
finally:
    ray.shutdown()
