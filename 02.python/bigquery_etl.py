
# ==== GCP Env. ====
import pandas as pd

import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage
import pandas_gbq


credentials, project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

bqclient = bigquery.Client(credentials=credentials, project=project_id,)
bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)


# excute
def excute_query(sql):
    return bqclient.query(sql)


# select
def select_query(sql):
    return bqclient.query(sql).to_dataframe()


# insert
def insert_query(tbl_nm, df) : 
    # get talbe
    origin = bqclient.query("select * from {}".format(tbl_nm)).to_dataframe() 
    # merge data
    df = pd.concat([origin, df])    
    # drop duplicate
    df = df.drop_duplicates()    
    # insert into tbl_nm
    return df.to_gbq(tbl_nm,
                     project_id = project_id,
                     if_exists='replace', # fail / replace / append
                    )

# insert append
def insert_append_query(tbl_nm, df) : 
    # insert into tbl_nm
    return df.to_gbq(tbl_nm,
                     project_id = project_id,
                     if_exists='append', # fail / replace / append
                    )


# insert
def to_bigquery(tbl_nm, df) :
    print(project_id, tbl_nm)
    return df.to_gbq(tbl_nm,
                     project_id = project_id,
                     if_exists='replace', # fail / replace / append
                    )

# merge
'''
    merge `taxono''my.tmp_hdly` as T
    using (
        select 'small' as checkword, 0 as is_cmpl_fc
      ) as R
    on T.checkword = R.checkword
    when not matched then
        insert row
'''