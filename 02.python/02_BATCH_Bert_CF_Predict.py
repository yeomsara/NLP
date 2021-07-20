import sys
sys.path.append('/home/ez-flow/big_data/python/')
import bigquery_etl as bq
import confusion_matrix_customized as cm_customize
import matplotlib.pyplot as plt
import bigquery_sql_load as sql_loader
import seaborn as sns
import pandas as pd
import numpy  as np
import torch
from tqdm.notebook import tqdm
from   datetime    import datetime
import pytz
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random
# Coding Env.
import warnings
from pandas.core.common import SettingWithCopyWarning

# warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")

def convert_lowercase(df):
    df_1 =  df.apply(lambda x: x.astype(str).str.lower() if(x.dtype == 'object') else x)
    upper_list = ['reviewId','asin','size','cmpl_fc1_cd']
    cols = list(set(upper_list)& set(df_1.columns))
    df_1[cols] = df_1[cols].apply(lambda x: x.astype(str).str.upper() if(x.dtype == 'object') else x)
    return df_1

def convert_uppercase(df):
    upper_list = ['reviewId','asin','sku']
    cols = list(set(upper_list)& set(df.columns))
    df[cols] = df[cols].apply(lambda x: x.astype(str).str.upper() if(x.dtype == 'object') else x)
    return df

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

## input data Embeding for bert 
def make_bert_input_embeding(input_df,x_cols):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
    print(input_df.shape)
    encoded_data = tokenizer.batch_encode_plus(
        input_df[x_cols].values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=256, 
        return_tensors='pt'
    )
    input_ids_train = encoded_data['input_ids']
    attention_masks_train = encoded_data['attention_mask']
    
    dataset = TensorDataset(input_ids_train, attention_masks_train)
    dataset_loader = DataLoader(dataset, 
                                   sampler=SequentialSampler(dataset), 
                                   batch_size=batch_size)
    return dataset_loader
    
## predict bert Model
def predict_bert(model,dataloader):
    model.eval()
    
    predictions, true_vals = [], []
    
    for batch in dataloader:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        # logits nomalized (log Probability -> softmax probability)
        logits = torch.nn.functional.softmax(outputs[0],dim=-1)
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    predictions = np.concatenate(predictions, axis=0)
            
    return predictions

def make_regidate(regi_df):
    regidate     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    regi_df['regidate'] = regidate
    regi_df['regidate'] = pd.to_datetime(regi_df['regidate'])
    return regi_df

## Convert Predict Class complain factor type
def convert_predict_class(reviewId_df,preds):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    print(label_dict_inverse)
    second_flat = np.array(preds).argpartition(-2)[:,-2]
    third_flat  = np.array(preds).argpartition(-3)[:,-3]
    preds_flat  = np.argmax(np.array(preds), axis=1).flatten()
    preds['y_pred'] = preds_flat
    preds['second_pred'] = second_flat
    preds['third_pred'] = third_flat
    preds['y_pred_class'] = preds_flat
    preds['second_class'] = second_flat
    preds['third_class'] = third_flat
    preds = preds.replace({'y_pred_class':label_dict_inverse,
                           'second_class':label_dict_inverse,
                           'third_class':label_dict_inverse})
    preds['reviewId'] = reviewId_df['reviewId'].tolist()
    return preds



##### Pipeline Start ######
try:
    ## (Setp 1-1) Load Data
    bert_train_sql,start_date,end_date,start_ym,end_ym   = sql_loader.review_input_load_sql()

    print(f'''==================================================================================''')
    print(f''' bert System target initialTime {end_ym} '{end_date}'(batch time)''')
    print(f''' bert System date yearmonth     between '{start_ym}' and '{end_ym}' ''')      
    print(f'''==================================================================================''')
    print(' (Setp 1-1) Load Data')
    df                = convert_lowercase(bq.select_query(bert_train_sql))
    df_1 = make_anal_df(df,0)
    print('           Original review_data        : ' ,df.shape)
    print(f'           rating(1~2) negative review : {df_1.shape}')
    print(f'           yeamonth new negative review_data : \n' ,df_1.groupby('yearmonth')['reviewId'].count().reset_index())
    test_df     = df_1.drop_duplicates('reviewId')
    test_df     = test_df[['reviewId','review_text']]
    reviewId_df = test_df[['reviewId']]

    ##(Step 1-2) Load complain factor priority and label
    ## make Label Dictionary (priority-> class number)
    label_dict = {  'recovery'        : 0, 
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


    ## (Step 2-1) Embeding Input DF
    print(' (Step 2-1) Embeding Input DF')
    batch_size  = 5
    input_x     = make_bert_input_embeding(test_df,'review_text')
    model       = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    device = torch.device("cpu")
    model.to(device)
    ## bert model directory
    model_dir = '/home/ez-flow/big_data/model/CF_Bert_Operation.model'
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))


    ## (Step 3-1) Predict Bert Model 
    print(' (Step 3-1) Predict Bert Model ')
    test_predictions = predict_bert(model,input_x)
    predict_df       = pd.DataFrame(test_predictions)
    predict_df.columns = list(label_dict)
    test_y_pred = convert_predict_class(reviewId_df,predict_df)


    ## (Step 4-1) Make Output DataFrame
    print(' (Step 4-1) Make Output DataFrame ')
    test_y_pred = test_y_pred.reindex(columns=['reviewId','y_pred','second_pred','third_pred','y_pred_class','second_class','third_class',
                                               'recovery', 'durability', 'defect', 'too soft', 'too hard',
                                               'missing parts', 'odor', 'sound', 'uncomfortable', 'size issue',
                                               'shipping damage', 'delivery', 'fiberglass', 'hard to set up',
                                               'slipping', 'cover issue', 'customer service', 'springs felt',
                                               'overall quality', 'no support','structure design', 'customer error', 'others'])
    test_y_pred['cum_prob_top3'] = [test_predictions[i,test_y_pred[['y_pred','second_pred','third_pred']].to_numpy()[i,:]].sum() for i in range(0,test_y_pred.shape[0])]
    test_y_pred['y_pred_prob'] = [test_predictions[i,test_y_pred[['y_pred','second_pred','third_pred']].to_numpy()[i,:]].tolist() for i in range(0,test_y_pred.shape[0])]

    label_cmpl_cf_dict = {  'recovery'         : 'CF011', 
                            'durability'       : 'CF001',
                            'defect'           : 'CF002', 
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

    label_dict_inverse = {v: k for k, v in label_dict.items()}
    bert_result = test_y_pred.rename(columns=label_cmpl_cf_dict)


    bert_result['y_pred_list']  = bert_result[['y_pred_class','second_class','third_class']].to_numpy().tolist()
    bert_result = pd.merge(bert_result,df_1[['reviewId','asin','date','yearmonth']].drop_duplicates(),how='left',on=['reviewId'])
    bert_result = bert_result.reindex(columns=['reviewId', 'asin', 'date', 'yearmonth', 'y_pred', 'second_pred', 'third_pred', 'y_pred_class',
                                                 'second_class', 'third_class', 'CF011', 'CF001', 'CF002', 'CF013',
                                                 'CF012', 'CF003', 'CF017', 'CF004', 'CF014', 'CF008', 'CF018', 'CF019',
                                                 'CF007', 'CF010', 'CF009', 'CF006', 'CF021', 'CF016', 'CF005', 'CF015',
                                                 'CF022', 'CF020', 'CF999', 'cum_prob_top3', 'y_pred_prob','y_pred_list'])
    
    reviewId_list = "','".join(bert_result.reviewId.unique())
    ## avoid data duplicates upload delete reviewId and upload
    

    bert_result = make_regidate(bert_result)
    # bert result db upload
    print(' (Step 5-1) connection DataBase ')
    bert_tbl_name = 'taxonomy.bert_cf1_predict_result'
    bq.excute_query(f''' DELETE FROM {bert_tbl_name} WHERE reviewId in({"'"+reviewId_list+"'"}) ''')
    print(f''' (Step 5-2) delete '{bert_tbl_name}' target reviewId count {len(bert_result.reviewId.unique())} ''')
    
    bq.insert_append_query(bert_tbl_name,bert_result)
    print(f'           >> Success Bert Predict Pipeline')
    print(f'           >> Success {bert_tbl_name} DataBase Upload')
except Exception as e :
    print(f'Bert Error : {e}')
