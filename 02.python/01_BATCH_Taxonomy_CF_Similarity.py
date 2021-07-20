import pandas as pd
import numpy  as np
import os
import re
import datetime
import pytz
import copy
from collections import Counter
from emoji   import UNICODE_EMOJI
from functools   import reduce
import sys
sys.path.append('/home/ez-flow/big_data/python/')
import confusion_matrix_customized as cm_customize
import bigquery_sql_load as sql_loader
import bigquery_etl as bq
import operator
import time
from IPython.display import display
# NLP Env.
import nltk
from nltk import FreqDist
from nltk.corpus   import stopwords
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer,LancasterStemmer
from nltk.corpus   import wordnet
from nltk.corpus   import sentiwordnet as swn
from nltk import sent_tokenize,word_tokenize,pos_tag
from sklearn.metrics import classification_report,plot_confusion_matrix,confusion_matrix,accuracy_score
# stop_words = stopwords.words('english')
import spacy
import gensim
from gensim import corpora
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import ray
import psutil
from tqdm.notebook import tqdm
from datetime import datetime
import pytz
#glove
from glove import Corpus,Glove

# Vis Env.
import pyLDAvis
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# GCP Env.
import google.auth
from google.cloud import bigquery
from googletrans import Translator
from google_trans_new import google_translator

# Coding Env.
import warnings
from pandas.core.common import SettingWithCopyWarning

credentials, project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

client = bigquery.Client(credentials=credentials, project=project_id )
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


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

def top20_df_brand(df):
    br_cat_rvw_rank  = pd.pivot_table(df, index = ['brand'], values = ['reviewId'], columns = ['prdct_ctgry_4_5'], aggfunc = ['count'], fill_value = 0, margins = True)#.reset_index()#.to_csv('ddd.csv')
    br_rvw_rank_all  = br_cat_rvw_rank['count']['reviewId']['All'].reset_index()
    br_rvw_rank_all  = br_rvw_rank_all.loc[(br_rvw_rank_all['brand'] != 'All'),]
    br_rvw_rank_all['rank'] = br_rvw_rank_all['All'].rank(ascending=False).astype(int)
    br_rvw_rank_all  = br_rvw_rank_all.sort_values(by='rank',ascending=True)
    br_rvw_rank_all  = br_rvw_rank_all[0:20]
    return br_rvw_rank_all['brand'].tolist()

#check multiprocessing progress 
def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])
        
def to_feather_df(df,name):
    df.reset_index().to_feather(f'temp/{name}.ftr')
    print(f'temp/{name}.ftr에 저장완료')
    
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


# (Step 2-1)grep complain factor3
def multiple_regex_list(text):
    regexList = [r"less then [0-9]+ day",r"less then [0-9] week",r"less then [0-9]+ month",r"less then [0-9] year",
                 r"for [0-9] day",r"for [0-9]+ week",r"for [0-9]+ month",r"for [0-9] year",'for a few week','for a few day','for a month day','for a year','for a month','for a day','for a week',
                 r"after [0-9]+ day",r"after [0-9]+ week",r"after [0-9]+ month",r"after [0-9] year",'a few months after','a few days after','a few years after','a few weeks after',
                 r"over [0-9]+ day",r"over [0-9]+ week",r"over [0-9]+ month",r"over [0-9] year",'within a year','within a week','within a day','within a year','within [0-9]+ month','within [0-9]+ day|within [0-9] week',
                 r'after a day',r'after a week',r'after a month',r'after a year','after one year','after one month','after [0-9]+ night',
                 r'have had it for about [0-9]+ day',r'have had it for about [0-9]+ month',r'have had it for about [0-9]+ week',r'have had it for about [0-9] year',
                 r'[0-9]+ days ago|[0-9]+ day ago',r'[0-9]+ week ago|[0-9]+ weeks ago',r'[0-9]+ month ago|[0-9]+ months ago',r'[0-9] year ago|[0-9] years ago']
    gotMatch = []
    for regex in regexList:
        s = list(set(re.findall(regex, text)))
        if len(s) > 0:
            gotMatch.append(' '.join(s))
    gotMatch = ','.join(gotMatch)
    return gotMatch

# (Step 2-2)Tokenized & make corpus & grep complain factor3
# @ray.remote
def tokenized_corpus(df):
    corpus    = []
    reviewid  = []
    category  = []
    cmpl_fc3_l  = []
    inch_list = []
    # if you wonder that nltk pos tag
    # nltk.help.upenn_tagset()
    N_POS_TAG   = ['CC','DT','EX','FW','LS','PDT','POS','PRP','PRP$','TO','WDT','WP','WRB']
    Y_POS_TAG   = ['JJ','JJR','JJS','MD','IN','NN','NNS','NNP','NNPS','RB','RBR','RBS','RP','UH','VB','VBG','VBD','VBN','VBP','VBZ']
    for i,v in enumerate(df['review_text']):
        try:
            inches   = re.findall(r'[0-9]+["]|[0-9]+inch|[0-9]+ inch',v)
            inches   = list(set(inches))
            cmpl_fc3 = multiple_regex_list(v)
            cmpl_fc3_l.append(cmpl_fc3)
            inch_list.append(inches)
            reviewid.append(str(df.iloc[i].reviewId)) 
            category.append(str(df.iloc[i].category)) 
            word = []
            for j in pos_tag(regexp_tokenize(v,"[\w']+")) :
#                 if (len(j[0])>2)  & (j[0].isascii()) & (j[1] in Y_POS_TAG) :
                if (j[0].isascii()) & (j[1] in Y_POS_TAG) :
                    word.append(j[0])
            mwe_tokenizer          = nltk.tokenize.MWETokenizer(mwe,separator=' ')
            tokenized_string       = [WordNetLemmatizer().lemmatize(x,pos='v') for x in word ]
            tokenized_string       = mwe_tokenizer.tokenize(word)
            tokenized_string       = [i for i in tokenized_string if i not in stopwords.words('english')]
#             tokenized_string       = [WordNetLemmatizer().lemmatize(x,pos='v') for x in tokenized_string ]
            corpus.append(tokenized_string)
        except : 
            pass
        corpus_df = pd.DataFrame({ 'corpus_list' : corpus,'reviewId' : reviewid,'category':category ,'cmpl_fc3':cmpl_fc3_l,'inch_regex':inch_list})
    return corpus_df

#initialization ray
num_logit_cpus = psutil.cpu_count()
ray.init(ignore_reinit_error=True,num_cpus=num_logit_cpus)

#(Step 3-1)Calculating Taxonomy similarity
@ray.remote
def taxonomy_similarity(cor_df):
    sim_result  = pd.DataFrame()
    for i in range(0,len(cor_df)):
        cat      =  cor_df.category.tolist()[i]
        corpus   =  cor_df.corpus_list.tolist()[i]
        reviewid =  cor_df.reviewId.tolist()[i]
        cat_cpl_fac   = cpl_factor[cpl_factor['category'].isin([cat,'all'])]
        cat_cpl_count = cat_cpl_fac.drop_duplicates(['cmpl_fc1','cmpl_fc2'],keep='first').groupby(['cmpl_fc1'])['cmpl_fc2'].count().reset_index()
        cat_cpl_count = cat_cpl_fac.groupby(['cmpl_fc1_cd','cmpl_fc1'])['cmpl_fc2'].count().reset_index()
        cat_cpl_count['cat_cf_ratio'] = cat_cpl_count['cmpl_fc2']/cat_cpl_count['cmpl_fc2'].sum()
        cat_cpl_count.columns         = ['cmpl_fc1_cd','cmpl_fc1','cat_cmpl_fc2_len','cat_cf1_ratio'] 
        cpl1_list   = cat_cpl_fac.cmpl_fc1.unique()
        cpl2_list   = cat_cpl_fac.cmpl_fc2.unique()
        for cf1 in cpl1_list : 
            cf1_intersec = set(corpus)&set([cf1])
            #if complain factor1 is in corpus
            if len(cf1_intersec) > 0:
                cf1_intersec = 0.5
                intersec     = list(set(corpus)&set([cf1]))
                cf2_df = pd.DataFrame({
                                          'category': cat,
                                          'reviewId': reviewid,
                                          'corpus'  : [corpus],
                                          'corpus_len': len(corpus),
                                          'cmpl_fc1': cf1,
                                          'cf1_intersect': cf1_intersec,
                                          'cmpl_fc2_len' : len(cat_cpl_fac[cat_cpl_fac['cmpl_fc1']==cf1].cmpl_fc2.unique()), 
                                          'cmpl_fc2': cf1,
                                          'synonym' : [None],
                                          'synonym_len': [0],
                                          'syn_intersect':'',
                                          'syn_match_len':[0]
                                     })
                cf2_df     = pd.merge(cf2_df,cat_cpl_count,how='left',on=['cmpl_fc1'])
                sim_result = pd.concat([cf2_df,sim_result])
            else :
                cf1_intersec = 0
            for cf2 in cat_cpl_fac[cat_cpl_fac['cmpl_fc1']==cf1].cmpl_fc2.unique() :
                syn_list    = [i for i in cat_cpl_fac[cat_cpl_fac['cmpl_fc2'] == cf2].synonym.unique() if i not in ['none',None]]
                lemma_list  = [i for i in cat_cpl_fac[cat_cpl_fac['cmpl_fc2'] == cf2].lemma.unique()  if i not in ['none',None]]
                syn_list = syn_list+lemma_list
                syn_list.append(cf2)
                intersec  = set(corpus)&set(syn_list)
                if ( len(intersec) > 0 ) :
                    cf2_df = pd.DataFrame({
                                          'category': cat,
                                          'reviewId': reviewid,
                                          'corpus'  : [corpus],
                                          'corpus_len':len(corpus),
                                          'cmpl_fc1': cf1,
                                          'cf1_intersect':cf1_intersec,
                                          'cmpl_fc2_len' :len(cat_cpl_fac[cat_cpl_fac['cmpl_fc1']==cf1].cmpl_fc2.unique()),                  
                                          'cmpl_fc2': cf2,
                                          'synonym' : [syn_list],
                                          'synonym_len'  : len(syn_list),
                                          'syn_intersect':','.join(list(intersec)),
                                          'syn_match_len':len(intersec)
                                          })
                    cf2_df     = pd.merge(cf2_df,cat_cpl_count,how='left',on=['cmpl_fc1'])
                    sim_result = pd.concat([cf2_df,sim_result])
    return sim_result

#(Step 4-1) Add taxonomy rules
def taxonomy_rule_chk(simi_rule_df, r_df):
    for i in r_df.index : 
        r_cd = r_df.loc[i, 'rule_code']
        cf1 = r_df.loc[i, 'cmpl_fc1']
        cf2 = r_df.loc[i, 'cmpl_fc2']
        cf3 = r_df.loc[i, 'cmpl_fc3']
        inch = r_df.loc[i, 'inch_len']
        rule1 = r_df.loc[i, 'rule1']
        rule2 = r_df.loc[i, 'rule2']
        rule3 = r_df.loc[i, 'rule3']
        
        if r_cd == 1 :
            # duration
            conditions = (simi_rule_df['cmpl_fc1'].isin([cf1])) & (~simi_rule_df['cmpl_fc3'].isin(['none','',' ']))
            simi_rule_df.loc[conditions, 'rule_check'] = 1
            simi_rule_df.loc[conditions, 'rule_score'] = 0.5
        elif r_cd == 2 :
            # inch
            conditions = (simi_rule_df['cmpl_fc1'].isin([cf1])) & (simi_rule_df['cmpl_fc2'].isin([cf2])) & (simi_rule_df['inch_rule']>=2)
            simi_rule_df.loc[conditions, 'rule_check'] = 2
            simi_rule_df.loc[conditions, 'rule_score'] = 0.5
        elif r_cd == 3 :
            # meta
            conditions = (simi_rule_df['cmpl_fc1'].isin([cf1])) & (simi_rule_df['cmpl_fc2'].isin([cf2])) & (simi_rule_df['corpus'].apply(lambda x : rule3 in x))
            simi_rule_df.loc[conditions, 'rule_check'] = 3
            simi_rule_df.loc[conditions, 'rule_score'] = 0.5
        elif r_cd == 4 :
            # filter
            conditions = (simi_rule_df['cmpl_fc1'].isin([cf1])) & (simi_rule_df['cmpl_fc2'].isin([cf2])) & (simi_rule_df['cmpl_fc2_list'].apply(lambda x : rule3 not in x))
            conditions = conditions | ( simi_rule_df['reviewId'].isin(simi_rule_df[conditions]['reviewId']) )
            simi_rule_df.loc[conditions, 'rule_check'] = 4
            simi_rule_df.loc[conditions, 'rule_score'] = -1
        else : 
            pass
            
        print("Rule_Code Number is : ", int(r_cd), "\t//\tCF1", cf1, " :",conditions.sum())
    
    return simi_rule_df

#(Step 5-1) Calulating Complain Factor 1 similarity 
def cal_similarity(similarity_df):
    key_cols = ['reviewId','asin','category','date', 'yearmonth', 'year', 'month']
    simi_max_df   = similarity_df.groupby(key_cols+['cmpl_fc1','cf1_intersect','cat_cf1_ratio'])['cmpl_fc2_len'].max().reset_index()
    simi_count_df = similarity_df.groupby(key_cols+['rule_score','cmpl_fc1','corpus_len'])['syn_match_len'].sum().reset_index()
    simi_test     = pd.merge(simi_max_df,simi_count_df,how='left',on=key_cols+['cmpl_fc1'])
    simi_test     = pd.merge(simi_test,priority_cf_dic[['cmpl_fc1','weight']],how='left',on=['cmpl_fc1'])
    simi_test['cf2_similarity']    = (simi_test['syn_match_len'] / simi_test['cmpl_fc2_len'])#*simi_test['cat_cf1_ratio']
    simi_test['corpus_similarity'] = (simi_test['syn_match_len'] / simi_test['corpus_len'] )
    simi_test['similarity']        = simi_test[['cf1_intersect','cf2_similarity','cat_cf1_ratio','corpus_similarity','rule_score','weight']].sum(axis=1)
    simi_test['rank'] = simi_test.groupby(['reviewId','category'])['similarity'].rank(ascending=False)
    simi_test         = simi_test.sort_values(['reviewId','rank'],ascending=[True,True])
    simi_test = simi_test.reindex(columns=key_cols+['corpus_len','cmpl_fc1','cmpl_fc2_len','syn_match_len','cf1_intersect',\
                                           'cat_cf1_ratio','cf2_similarity','corpus_similarity','rule_score','weight','similarity','rank'])
    return simi_test


##### Pipeline Start ######
try:
    ##(P1) Load input sql  
    ##load AMZ reviews sql
    review_sql,start_date,end_date,start_ym,end_ym    = sql_loader.review_input_load_sql()
    ##########################################
    # sql_cd 1 == stopword sql     
    # sql_cd 2 == complain Factor sql     
    # sql_cd 3 == bsr_brnad sql     
    # sql_cd 4 == part_sql sql    
    # sql_cd 5 == taxonomy rule sql 
    ##########################################
    filter_sql    = sql_loader.load_sql(1)
    factor_sql    = sql_loader.load_sql(2)
    bsr_brand_sql = sql_loader.load_sql(3)
    part_sql      = sql_loader.load_sql(4)
    rules_sql     = sql_loader.load_sql(5)
    print(f'''==================================================================================''')
    print(f''' Taxonomy System target initialTime {end_ym} '{end_date}'(batch time) ''')
    print(f''' Taxonomy System date yearmonth     between '{start_ym}' and '{end_ym}' ''')      
    print(f'''==================================================================================''')
    print(' (Setp 1-1) Load complain factor dataframe & multi word express')
    ##(P2) Load complain factor dataframe & multi word express
    cpl_factor    = convert_lowercase(bq.select_query(factor_sql))
    cpl_factor['lemma'] = cpl_factor['synonym'].apply(lambda x : lemmatize(x))
    cpl_factor['porter_stem'] = cpl_factor['synonym'].apply(lambda x : PorterStemmer().stem(x))
    cmpl_fc_list  = list(set(cpl_factor['cmpl_fc1'].unique().tolist())|set(cpl_factor['cmpl_fc2'].unique().tolist())|set(cpl_factor['synonym'].unique().tolist())\
                         |set(cpl_factor['lemma'].unique().tolist()))
    multi_express = list(filter(lambda x: len(x.split(' '))>1 , cmpl_fc_list))
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
    parts_list.sort(reverse=True)
    stop_words     = stopword_df['stopword'].tolist()
    rules_df       = convert_lowercase(bq.select_query(rules_sql))
    values = {'cmpl_fc2':'',"cmpl_fc3": 0, "inch_len": 0, "rule1": 0, "rule2": 0,'rule3':''}
    rules_df = rules_df.fillna(value=values)
    

    ##(P4) Make input dataframe 
    neg_df = make_anal_df(df,0)
    print('           complain factor             : ' ,cpl_factor.shape)
    print(f'           all complain factor         : {len(cmpl_fc_list)}')
    print(f'           multi_express lenth         : {len(mwe)}')
    print('           stop_words                  : ' ,len(stop_words))
    print('           bsr_brand_sql               : ' ,bsr_brand_df.shape)
    print('           parts_df                    : ' ,parts_df.shape)
    print('           Taxonomy rules_df           : ' ,rules_df.shape)
    print('           Original review_data        : ' ,df.shape)
    print(f'           rating(1~2) negative review : {neg_df.shape}')
    print(f'           yeamonth new negative review_data : \n' ,neg_df.groupby('yearmonth')['reviewId'].count().reset_index())
    
    
    print(' (Setp 2-1) Tokenized & make corpus ')
    ##(P5) Tokenized & make corpus 
    neg_df = neg_df.drop_duplicates('reviewId')
    corpus_df = tokenized_corpus(neg_df)
    corpus_df['inch_rule'] = corpus_df['inch_regex'].apply(lambda x : len(x))

    print(' (Setp 3-1) Calculating Taxonomy similarity (CF1-CF2) ')
    ##(P6) Calculating Taxonomy similarity (CF1-CF2)
    category_cols  = ['reviewId','asin', 'date', 'yearmonth', 'year', 'month']
    similarity_df2 = [taxonomy_similarity.remote(corpus_df[corpus_df['category']==i]) for i in corpus_df['category'].unique()]
#     similarity_df2 = ray_multiprocessing_progress(similarity_df2)
    similarity_df2 = pd.concat(ray.get(similarity_df2))
#     similarity_df2 = taxonomy_similarity(corpus_df)
    similarity_df2 = similarity_df2.drop_duplicates(['reviewId','cmpl_fc1','cmpl_fc2']).sort_values(['reviewId','cmpl_fc1'],ascending=[True,True])
    similarity_df  = pd.merge(similarity_df2,neg_df[category_cols],how='left',on=['reviewId'])
    similarity_df  = pd.merge(similarity_df,corpus_df[['reviewId','cmpl_fc3','inch_regex','inch_rule']],how='left',on=['reviewId'])
    similarity_df['rule_check'] = 0
    similarity_df['rule_score'] = 0

    print(' (Setp 4-1) Reflect to taxonomy rules ')
    ##(P7) Reflect to taxonomy rules
    similarity_rule_df    = similarity_df[similarity_df['cmpl_fc1'].isin(rules_df.cmpl_fc1.unique().tolist())]
    similarity_no_rule_df = similarity_df[~similarity_df['cmpl_fc1'].isin(rules_df.cmpl_fc1.unique().tolist())]

    similarity_rule_check_df = similarity_rule_df.copy()
    similarity_rule_check_df['rule_score'] = 0

    cf2_list_df = similarity_rule_check_df.groupby(['reviewId','cmpl_fc1'])['cmpl_fc2'].apply(list).reset_index()
    cf2_list_df.rename(columns={"cmpl_fc2":"cmpl_fc2_list"}, inplace=True)

    similarity_rule_check_df = similarity_rule_check_df.merge(cf2_list_df[['reviewId', 'cmpl_fc2_list']], how='left', on='reviewId')
    similarity_rule_check_df = taxonomy_rule_chk(similarity_rule_check_df, rules_df)
    print("           Befoe Rule Filtering : ", similarity_rule_check_df.shape)
    similarity_rule_check_df = similarity_rule_check_df[similarity_rule_check_df['rule_score']>0]
    print("           After Rule Filtering : ", similarity_rule_check_df.shape)
    final_similarity_rule_df = pd.concat([similarity_rule_check_df, similarity_no_rule_df])

    print("           Merge                : ", final_similarity_rule_df.shape)
    final_similarity_rule_df = final_similarity_rule_df.fillna(0)

    print(' (Setp 5-1) complain Factor2 db upload ')
    ## complain Factor2 db upload
    cf2_simi_result = final_similarity_rule_df[[ 'reviewId','asin','category','date', 'yearmonth', 'year', 'month',
                                                 'corpus_len', 'cmpl_fc1', 'cf1_intersect','cmpl_fc2_len', 'cmpl_fc2', 'cmpl_fc3', 'inch_rule', 'syn_intersect',
                                                 'syn_match_len', 'cmpl_fc1_cd', 'cat_cmpl_fc2_len', 'cat_cf1_ratio','rule_score']]
    
    cf2_simi_result = cf2_simi_result.drop_duplicates()
    cf2_simi_result = make_regidate(cf2_simi_result)
    cf2_tbl_name = 'taxonomy.taxonomy_cf2_result'
    reviewId_list = "','".join(cf2_simi_result.reviewId.unique())
    ## avoid data duplicates upload delete reviewId and upload
    bq.excute_query(f''' DELETE FROM {cf2_tbl_name} WHERE reviewId in({"'"+reviewId_list+"'"}) ''')
    print(f'''           DELETE  '{cf2_tbl_name}' table target yearmonth between '{start_ym}' and '{end_ym}' target reviewId count {len(cf2_simi_result.reviewId.unique())}  ''')
    bq.insert_append_query(cf2_tbl_name,cf2_simi_result)
    print(f'          >> Success {cf2_tbl_name} DataBase Upload')
    
    print(' (Setp 6-1) Calculate Priority score by Complain Factor1  ')
    ##(P8) Priority Complain Factor 1 
    y_cf_dic   = {  'recovery'        : 0, 
                    'durability'      : 1,
                    'defect'          : 2, 
                    'too hard'        : 3,  
                    'too soft'        : 4, 
                    'missing parts'   : 5,
                    'odor'            : 6,
                    'sound'           : 7,
                    'uncomfortable'   : 8,
                    'size issue'      : 9,
                    'shipping damage' : 10,
                    'delivery'        : 11,
                    'fiberglass'      : 12,
                    'hard to set up'  : 13, 
                    'slipping'        : 14, 
                    'cover issue'     : 15, 
                    'customer service': 16, 
                    'springs felt'    : 17,
                    'overall quality' : 18,
                    'no support'      : 19,
                    'customer error'  : 20,
                    'structure design': 21,
                    'others'          : 22, 
               }

    priority_cf_dic = pd.DataFrame({'cmpl_fc1': list(y_cf_dic),
                       'priority' : range(1,len(y_cf_dic)+1)})
    priority_cf_dic['weight'] = round((1/priority_cf_dic['priority']),2)

    print(' (Setp 7-1) ) Calculating final complain factor  ')
    ##(P9) Calculating final complain factor 
    simi_result      = cal_similarity(final_similarity_rule_df)
    print(' (Setp 8-1) ) Similarity normalization  through Softmax function  ')
    ##(P10) Similarity normalization  through Softmax function
    simi_result['exp_similarity']         = simi_result.groupby(['reviewId','yearmonth'])['similarity'].apply(np.exp)
    simi_result['SUM_EXP_PROB']           = simi_result.groupby(['reviewId','yearmonth'])['exp_similarity'].transform('sum')
    simi_result['softmax_similarity']     = simi_result['exp_similarity'] / simi_result['SUM_EXP_PROB']
    simi_result                      = simi_result.sort_values(['yearmonth','reviewId','similarity'],ascending=[False,True,False])
    simi_result['rank']              = simi_result.groupby(['reviewId','yearmonth'])['softmax_similarity'].rank(ascending=False)
    simi_result['cumsum_similarity'] = simi_result.groupby(['reviewId','yearmonth'])['softmax_similarity'].cumsum()
    simi_result = simi_result.drop(columns=['exp_similarity','SUM_EXP_PROB'])
    simi_result      = simi_result.sort_values(['reviewId','rank'],ascending=[True,True])
    simi_result = simi_result.reindex(columns=['reviewId', 'asin', 'category','date', 'yearmonth', 'year', 'month',  
                                               'corpus_len', 'cmpl_fc1','cmpl_fc2_len', 'syn_match_len', 'cf1_intersect', 'cat_cf1_ratio',
                                               'cf2_similarity', 'corpus_similarity', 'rule_score', 'weight',
                                               'similarity', 'rank', 'softmax_similarity', 'cumsum_similarity'])
    simi_result = simi_result.drop_duplicates()
    simi_result = make_regidate(simi_result)
    # complain Factor1 db upload
    print(' (Setp 9-1)  complain Factor1 db upload ')
    cf1_tbl_name = 'taxonomy.taxonomy_cf1_result'
    reviewId_list = "','".join(simi_result.reviewId.unique())
    ## avoid data duplicates upload delete reviewId and upload
    bq.excute_query(f''' DELETE FROM {cf1_tbl_name} WHERE reviewId in({"'"+reviewId_list+"'"}) ''')
    print(f''' Upload New reviews count : {len(reviewId_list)} ''')
    print(f'''           DELETE  '{cf1_tbl_name}' table target yearmonth between '{start_ym}' and '{end_ym}' target reviewId count {len(simi_result.reviewId.unique())}  ''')
    bq.insert_append_query(cf1_tbl_name,simi_result)
    print(f'           >> Success Taxonomy Pipeline')
    print(f'''           >>Success dataframe({simi_result.shape}) '{cf1_tbl_name}' DataBase Upload''')
except Exception as e:
    print(f'Error : {e}')
finally :
    ray.shutdown()
