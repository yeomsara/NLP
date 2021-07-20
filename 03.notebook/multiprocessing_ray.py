def taxonomy_similarity(cat,br,corpus,text,reviewid):
    cat_cpl_fac = cpl_factor[cpl_factor['category'].isin([cat,'all'])]
    cat_cpl_count = cat_cpl_fac.groupby(['cmpl_fc1_cd','cmpl_fc1'])['cmpl_fc2'].count().reset_index()
    cat_cpl_count['cat_cf_ratio'] = cat_cpl_count['cmpl_fc2']/cat_cpl_count['cmpl_fc2'].sum()
    cat_cpl_count.columns = ['cmpl_fc1_cd','cmpl_fc1','cat_cmpl_fc2_len','cat_cf1_ratio']    
    cpl1_list   = cat_cpl_fac.cmpl_fc1.unique()
    fac_list    = []
    sim_result  = pd.DataFrame()
    for cf1 in cpl1_list : 
        cf1_intersec = set(corpus)&set([cf1])
        if len(cf1_intersec) > 0:
            cf1_intersec = 0.5
            intersec  = list(set(corpus)&set([cf1]))
            cf2_df = pd.DataFrame({
                                  'category':cat,
                                  'brand':br,
                                  'reviewId':reviewid,
                                  'corpus':[corpus],
                                  'corpus_len':len(corpus),
                                  'cmpl_fc1':cf1,
                                  'cf1_intersect':cf1_intersec,
                                  'cmpl_fc2_len':len(cat_cpl_fac[cat_cpl_fac['cmpl_fc1']==cf1].cmpl_fc2.unique()), 
                                  'cmpl_fc2':[cf1],
                                  'synonym' : [None],
                                  'synonym_len': [0],
                                  'syn_intersect ':[None],
                                  'syn_match_len':[0]
                                 })
            cf2_df = pd.merge(cf2_df,cat_cpl_count,how='left',on=['cmpl_fc1'])
            sim_result = pd.concat([cf2_df,sim_result])
        else :
            cf1_intersec = 0
        for cf2 in cat_cpl_fac[cat_cpl_fac['cmpl_fc1']==cf1].cmpl_fc2.unique() :
            syn_list  = [i for i in cat_cpl_fac[cat_cpl_fac['cmpl_fc2'] == cf2].synonym.unique() if i not in ['none',None]]
            syn_list.append(cf2)
            intersec  = set(corpus)&set(syn_list)
            match_len = len(re.findall(r' %s'%(cf2), text))
            if ( len(intersec) > 0 ) :
                cf2_df = pd.DataFrame({
                                  'category':cat,
                                  'brand':br,
                                  'reviewId':reviewid,
                                  'corpus':[corpus],
                                  'corpus_len':len(corpus),
                                  'cmpl_fc1':cf1,
                                  'cf1_intersect':cf1_intersec,
                                  'cmpl_fc2_len':len(cat_cpl_fac[cat_cpl_fac['cmpl_fc1']==cf1].cmpl_fc2.unique()),                  
                                  'cmpl_fc2':cf2,
                                  'synonym' : [syn_list],
                                  'synonym_len': len(syn_list),
                                  'syn_intersect ':[list(intersec)],
                                  'syn_match_len':len(intersec)
                                 })
#             print(f'========================================================================================')
#             print(f'▶ reviewId : {reviewid} ')
#             print(f'▶ text : {text}')
#             print(f'========================================================================================')
#             print(f'▶corpus : {corpus}')
#             print(f'========================================================================================')
#             print(f' complain factor1 : {cf1}')
#             print(f'  → cf1 word : {cf1_intersec}')
#             print(f'       compalin factor2 : {cf2}')
#             print(f'           synonym list :  {syn_list}')
#             print(f'              intersect : ( {intersec} )')
#             print(f'              ■ match_len : {match_len} / intersection : {intersec} ')
                cf2_df = pd.merge(cf2_df,cat_cpl_count,how='left',on=['cmpl_fc1'])
                sim_result = pd.concat([cf2_df,sim_result])
    return sim_result

@ray.remote
def chunk_corpus(cor_df):
#     print(f'brand : {cor_df.brand.unique()} / shape : {cor_df.shape}')
    similarity_df  = pd.DataFrame()
    for i in range(0,len(cor_df)):
        cat =  cor_df.category.tolist()[i]
        br  =  cor_df.brand.tolist()[i]
        cor =  cor_df.corpus_list.tolist()[i]
        reviewId = cor_df.reviewId.tolist()[i]
        text = cor_df.review_text.tolist()[i]
        ss = taxonomy_similarity(cat,br,cor,text,reviewId)
        similarity_df = pd.concat([similarity_df,ss])
    return similarity_df

def ray_multiprocessing_progress(ray_df):
    for x in tqdm(to_iterator(ray_df), total=len(ray_df)):
        pass
    ray_df  = pd.concat(ray.get(ray_df))
    return ray_df 


similarity_df2 = [chunk_corpus.remote(corpus_df[corpus_df['brand']==i]) for i in tqdm(corpus_df['brand'].unique())]
similarity_df2 = ray_multiprocessing_progress(similarity_df2)

# similarity_df  = pd.DataFrame()
# for i in range(0,50):
#     cat =  corpus_df.category.tolist()[i]
#     br  =  corpus_df.brand.tolist()[i]
#     cor =  corpus_df.corpus_list.tolist()[i]
#     reviewId = corpus_df.reviewId.tolist()[i]
#     text = corpus_df.review_text.tolist()[i]
#     ss = taxonomy_similarity(cat,br,cor,text,reviewId)
#     similarity_df = pd.concat([similarity_df,ss])