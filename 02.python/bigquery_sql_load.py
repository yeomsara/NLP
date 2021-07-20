
def review_input_load_sql():
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    YM = (datetime.now() + relativedelta(months=-1)).strftime('%Y-%m')
    review_sql = f'''WITH base_table AS (SELECT A.*,B.prdct_ctgry_4_5,B.brand,RANK() OVER(partition by A.reviewId order by crawlTime desc) as rank
                                FROM `market-analysis-project-91130`.crwl.amz_rvw_all A 
                                LEFT JOIN `market-analysis-project-91130`.meta.crwl_amz_pdt_mst_all B ON A.asin =B.asin)
                    SELECT  A.reviewId,
                            A.region,		
                            A.prdct_ctgry_4_5 as category,
                            A.asin,
                            date(A.date)as date,
                            A.rating,
                            A.title,
                            lower(A.review_text) as review_text
                    FROM base_table A 
                    WHERE A.prdct_ctgry_4_5 IN ('Beds','Bed Frames','Mattresses','Box Springs','Sofas & Couches','Mattress Toppers')
                    and rank = 1
                    AND A.reviewId not in (select reviewId
                                              from (
                                                     SELECT reviewId,count(DISTINCT prdct_ctgry_4_5)
                                                     FROM base_table
                                                     GROUP BY reviewId 
                                                     HAVING count(DISTINCT prdct_ctgry_4_5) > 1
                                                   )
                                          )		
                    AND substring(A.date,1,7) = '{YM}'
                    and A.brand is not null
                    order by brand,A.prdct_ctgry_4_5,reviewId,date 
            '''
    return review_sql,YM


def load_sql(sql_cd):
    # sql_cd 1 == stopword sql     
    # sql_cd 2 == complain Factor sql     
    # sql_cd 3 == bsr_brnad sql     
    # sql_cd 4 == part_sql sql    
    # sql_cd 5 == taxonomy rule sql
    # sql_cd 6 == shiny Factor
    if sql_cd == 1 :
        sql =f''' 
                    SELECT distinct stopword 
                    FROM market-analysis-project-91130.taxonomy.stp_wds;
              '''
    elif sql_cd == 2:
        sql = '''
                    SELECT distinct T2.category,T1.*
                    FROM (SELECT a.cmpl_fc1,a.cmpl_fc1_cd,b.cmpl_fc2,c.synonym,c.lemma
                          FROM market-analysis-project-91130.taxonomy.cmpl_fc1_dic a
                          FULL OUTER JOIN market-analysis-project-91130.taxonomy.cmpl_fc2_dic b     on lower(a.cmpl_fc1_cd) = lower(b.cmpl_fc1_cd)
                          FULL OUTER JOIN market-analysis-project-91130.taxonomy.cmpl_fc2_syn_dic c on lower(b.cmpl_fc2) = lower(c.cmpl_fc2)
                    ) T1
                    LEFT JOIN market-analysis-project-91130.taxonomy.pdt_ctg T2 ON   T1.cmpl_fc1_cd = T2.cmpl_fc1_cd
              '''
    elif sql_cd == 3:
        sql =  '''
                    SELECT DISTINCT *
                    FROM (
                        SELECT
                            t1.category as prdct_ctgry_4_5 ,
                            LOWER(t1.brand) as brand ,
                            t1.rank ,
                            CASE
                                WHEN t2.brand_adj IS NOT NULL THEN lower(t2.brand_adj)
                                ELSE lower(t1.brand)
                            END AS brand_adj
                        FROM
                            (
                            SELECT
                                bsr_ctgry as Category,
                                lower(brand) as brand,
                                min(brand_adj_rank) as rank
                            FROM
                                rvw_mst.bsr_top10_brand
                            WHERE
                                bsr_ctgry IN ('Beds','Bed Frames','Mattresses','Box Springs','Sofas & Couches','Mattress Toppers')
                            GROUP BY
                                bsr_ctgry,
                                lower(brand) ) t1
                        LEFT JOIN `market-analysis-project-91130`.meta.brand_mapping t2 on
                            t1.brand = lower(t2.brand)
                    )
                    ORDER BY
                            1,
                            3
               '''
    elif sql_cd == 4:
        sql =  '''
                  SELECT *
                  FROM market-analysis-project-91130.taxonomy.prt_wds
                  ORDER BY 1 desc
               '''
    elif sql_cd == 5:
        sql = '''
                    SELECT *
                    FROM taxonomy.taxonomy_rule
                    '''
    elif sql_cd == 6:
        sql = '''
                    SELECT *
                    FROM taxonomy.shn_kwd 
              '''
    else :
        print("등록된 sql cd 가 아닙니다.")
    return sql


def Hybrid_Model_input():
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    YM = (datetime.now() + relativedelta(months=-1)).strftime('%Y%m')
    taxonomy_sql = f'''SELECT  *
                   FROM    taxonomy.taxonomy_cf1_result
                   WHERE   yearmonth  = '{YM}'  '''
    
    bert_sql =f'''SELECT  *
                   FROM    taxonomy.bert_cf1_predict_result
                   WHERE   yearmonth  = '{YM}'  '''
    return bert_sql,taxonomy_sql,YM


def monthly_asin_cat_master():
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    YM = (datetime.now() + relativedelta(months=-1)).strftime('%Y-%m')
    category_sql =f'''
                   WITH base_table AS (SELECT A.*,B.prdct_ctgry_4_5,B.brand,RANK() OVER(partition by A.reviewId order by crawlTime desc) as rank
                    FROM `market-analysis-project-91130`.crwl.amz_rvw_all A 
                    LEFT JOIN `market-analysis-project-91130`.meta.crwl_amz_pdt_mst_all B ON A.asin =B.asin)
                    SELECT DISTINCT 
                           t1.reviewId,
                           t1.review_text,
                           t1.asin,
                           CASE WHEN lower(t3.brand_adj) IS NOT NULL
                                 THEN lower(t3.brand_adj)
                                 ELSE lower(t1.brand)
                                 END as brand,
                           t3.brand_adj_rank  as brand_rank,
                           t1.prdct_ctgry_4_5 as category,
                           t2.category_1 ,
                           t2.category_2 ,
                           t2.category_3 ,
                           t2.inch,
                           t2.inch_adj,
                           t2.sub_size
                    from base_table t1
                    LEFT JOIN meta.crwl_amz_stck_pdt_master_top20_total t2 ON t1.asin = t2.retailersku 
                    LEFT JOIN rvw_mst.bsr_top10_brand  t3 on lower(t1.brand) = lower(t3.brand)  and lower(t1.prdct_ctgry_4_5) = lower(t3.bsr_ctgry) and t3.chnnl_nm ='US AMZ'
                    WHERE substring(t1.date,1,7)  = '{YM}'
                    '''
    return category_sql



def monthly_asin_cat_master():
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
#     YM = datetime.now().strftime('%Y-%m')
    YM = (datetime.now() + relativedelta(months=-1)).strftime('%Y-%m')
    category_sql =f'''
                   WITH base_table AS (SELECT A.*,B.prdct_ctgry_4_5,B.brand,RANK() OVER(partition by A.reviewId order by crawlTime desc) as rank
                    FROM `market-analysis-project-91130`.crwl.amz_rvw_all A 
                    LEFT JOIN `market-analysis-project-91130`.meta.crwl_amz_pdt_mst_all B ON A.asin =B.asin)
                    SELECT DISTINCT 
                           t1.reviewId,
                           t1.review_text,
                           t1.asin,
                           CASE WHEN lower(t3.brand_adj) IS NOT NULL
                                 THEN lower(t3.brand_adj)
                                 ELSE lower(t1.brand)
                                 END as brand,
                           t3.brand_adj_rank  as brand_rank,
                           t1.prdct_ctgry_4_5 as category,
                           t2.category_1 ,
                           t2.category_2 ,
                           t2.category_3 ,
                           t2.inch,
                           t2.inch_adj,
                           t2.sub_size
                    from base_table t1
                    LEFT JOIN meta.crwl_amz_stck_pdt_master_top20_total t2 ON t1.asin = t2.retailersku 
                    LEFT JOIN rvw_mst.bsr_top10_brand  t3 on lower(t1.brand) = lower(t3.brand)  and lower(t1.prdct_ctgry_4_5) = lower(t3.bsr_ctgry) and t3.chnnl_nm ='US AMZ'
                    WHERE substring(t1.date,1,7)  = '{YM}'
                    '''
    return category_sql


def Keyword_Trend_SQL():
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    start_ym = (datetime.now() + relativedelta(months=-13)).strftime('%Y%m')
    end_ym   = (datetime.now() + relativedelta(months=-1)).strftime('%Y%m')   
    negative = f'''
                SELECT *
                FROM taxonomy.negative_keyword_anal
                WHERE yearmonth between '{start_ym}' and '{end_ym}'
                '''
    positive = f'''
                SELECT *
                FROM taxonomy.positive_keyword_anal
                WHERE yearmonth between '{start_ym}' and '{end_ym}'
                '''
    return negative,positive,start_ym,end_ym