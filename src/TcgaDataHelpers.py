import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

from dataset import (
    AggCategoricalAnnotatedCellLineDataset,
    AggCategoricalAnnotatedPdxDataset,
    AggCategoricalAnnotatedTcgaDataset    
)

def get_pdx(drug_name, type_set):
    random.seed(42)
    np.random.seed(42)
    
    if type_set == 'test':
        is_train = False
    else:
        is_train = True
    if type_set == 'train_all':
        train_all = True
    else:
        train_all = False
    pdx_train = AggCategoricalAnnotatedPdxDataset(
        is_train=is_train, 
        only_cat_one_drugs=True
    )
    pdx_train_response_drug_df = pdx_train.pdx_response[pdx_train.pdx_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    pdx_train_mut_df = pdx_train.raw_mutations[pdx_train.raw_mutations.index.isin(pdx_train_response_drug_df.Sample)]
    pdx_train_merged_df = pd.merge(pdx_train_mut_df.reset_index(), pdx_train_response_drug_df.drop("drug_name", axis = 1), on = "Sample").set_index("Sample")

    if is_train and (len(pdx_train_merged_df) > 1) and not train_all:
        df_train, df_val = train_test_split(pdx_train_merged_df, test_size=0.2, random_state=42, stratify=pdx_train_merged_df['response'])
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(pdx_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for PDX {type_set} set')
    return pdx_train_merged_df

def get_tcga(drug_name, type_set):
    random.seed(42)
    np.random.seed(42)
    
    if type_set == 'test':
        is_train = False
    else:
        is_train = True
    if type_set == 'train_all':
        train_all = True
    else:
        train_all = False
    tcga_train = AggCategoricalAnnotatedTcgaDataset(
        is_train=is_train,
        only_cat_one_drugs=True,
    )
    tcga_train_response_drug_df = tcga_train.tcga_response[tcga_train.tcga_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    # tcga_train_response_drug_df = tcga_train.tcga_response # considering all train TCGA entities (if filter for specific drug (docetaxel) get empty tensor for loss1)
    tcga_train_mut_df = tcga_train.raw_mutations[tcga_train.raw_mutations.index.isin(tcga_train_response_drug_df.submitter_id)]
    tcga_train_merged_df = pd.merge(tcga_train_mut_df.reset_index(), tcga_train_response_drug_df.drop("drug_name", axis = 1), on = "submitter_id").set_index("submitter_id")
#     tcga_train_merged_df = tcga_train_mut_df
    if is_train and (len(tcga_train_merged_df) > 1) and not train_all:
        df_train, df_val = train_test_split(tcga_train_merged_df, test_size=0.2, random_state=42, stratify=tcga_train_merged_df['response'])
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(tcga_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for TCGA {type_set} set')
    return tcga_train_merged_df

def get_pdx_cnv(drug_name, type_set):
    random.seed(42)
    np.random.seed(42)
    
    if type_set == 'test':
        is_train = False
    else:
        is_train = True
    if type_set == 'train_all':
        train_all = True
    else:
        train_all = False
    pdx_train = AggCategoricalAnnotatedPdxDataset(
        is_train=is_train, 
        only_cat_one_drugs=True
    )
    pdx_train_response_drug_df = pdx_train.pdx_response[pdx_train.pdx_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    pdx_train_cnv_df = pdx_train.cnv[pdx_train.cnv.index.isin(pdx_train_response_drug_df.Sample)]
    pdx_train_merged_df = pd.merge(pdx_train_cnv_df.reset_index(), pdx_train_response_drug_df.drop("drug_name", axis = 1), on = "Sample").set_index("Sample")
#     print(pdx_train_merged_df.shape)
    if is_train and (len(pdx_train_merged_df) > 1) and not train_all:
        df_train, df_val = train_test_split(pdx_train_merged_df, test_size=0.2, random_state=42, stratify=pdx_train_merged_df['response'])
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(pdx_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for PDX {type_set} set')
    return pdx_train_merged_df

def get_tcga_cnv(drug_name, type_set):
    random.seed(42)
    np.random.seed(42)
    
    if type_set == 'test':
        is_train = False
    else:
        is_train = True
    if type_set == 'train_all':
        train_all = True
    else:
        train_all = False
    tcga_train = AggCategoricalAnnotatedTcgaDataset(
        is_train=is_train,
        only_cat_one_drugs=True,
    )
    tcga_train_response_drug_df = tcga_train.tcga_response[tcga_train.tcga_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    # tcga_train_response_drug_df = tcga_train.tcga_response # considering all train TCGA entities (if filter for specific drug (docetaxel) get empty tensor for loss1)
    tcga_train_cnv_df = tcga_train.cnv[tcga_train.cnv.index.isin(tcga_train_response_drug_df.submitter_id)]
    tcga_train_merged_df = pd.merge(tcga_train_cnv_df.reset_index(), tcga_train_response_drug_df.drop("drug_name", axis = 1), on = "submitter_id").set_index("submitter_id")
#     print(tcga_train_merged_df.shape)
    if is_train and (len(tcga_train_merged_df) > 1) and not train_all:
        df_train, df_val = train_test_split(tcga_train_merged_df, test_size=0.2, random_state=42, stratify=tcga_train_merged_df['response'])
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(tcga_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for TCGA {type_set} set')
    return tcga_train_merged_df

def get_pdx_mut_cnv(drug_name, type_set):
    random.seed(42)
    np.random.seed(42)
    
    if type_set == 'test':
        is_train = False
    else:
        is_train = True
    if type_set == 'train_all':
        train_all = True
    else:
        train_all = False
    pdx_train = AggCategoricalAnnotatedPdxDataset(
        is_train=is_train, 
        only_cat_one_drugs=True
    )
    pdx_train_response_drug_df = pdx_train.pdx_response[pdx_train.pdx_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    pdx_train_mut_df = pdx_train.raw_mutations[pdx_train.raw_mutations.index.isin(pdx_train_response_drug_df.Sample)]
    
    pdx_train_cnv_df = pdx_train.cnv[pdx_train.cnv.index.isin(pdx_train_response_drug_df.Sample)]
    pdx_train_merged_mut_cnv_df = pd.merge(pdx_train_mut_df.reset_index(), pdx_train_cnv_df.reset_index(), on = "Sample").set_index("Sample")
    pdx_train_merged_df = pd.merge(pdx_train_merged_mut_cnv_df.reset_index(), pdx_train_response_drug_df.drop("drug_name", axis = 1), on = "Sample").set_index("Sample")

#     pdx_train_merged_df = pd.merge(pdx_train_mut_df.reset_index(), pdx_train_response_drug_df.drop("drug_name", axis = 1), on = "Sample").set_index("Sample")
    
    if is_train and (len(pdx_train_merged_df) > 1) and not train_all:
#         print(pdx_train_merged_df.shape)
        df_train, df_val = train_test_split(pdx_train_merged_df, test_size=0.2, random_state=42, stratify=pdx_train_merged_df['response'])
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(pdx_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for PDX {type_set} set')
    return pdx_train_merged_df

def get_tcga_mut_cnv(drug_name, type_set):
    random.seed(42)
    np.random.seed(42)
    
    if type_set == 'test':
        is_train = False
    else:
        is_train = True
    if type_set == 'train_all':
        train_all = True
    else:
        train_all = False
    tcga_train = AggCategoricalAnnotatedTcgaDataset(
        is_train=is_train,
        only_cat_one_drugs=True,
    )
    tcga_train_response_drug_df = tcga_train.tcga_response[tcga_train.tcga_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    # tcga_train_response_drug_df = tcga_train.tcga_response # considering all train TCGA entities (if filter for specific drug (docetaxel) get empty tensor for loss1)
    tcga_train_mut_df = tcga_train.raw_mutations[tcga_train.raw_mutations.index.isin(tcga_train_response_drug_df.submitter_id)]
    
    tcga_train_cnv_df = tcga_train.cnv[tcga_train.cnv.index.isin(tcga_train_response_drug_df.submitter_id)]
    tcga_train_merged_mut_cnv_df = pd.merge(tcga_train_mut_df.reset_index(), tcga_train_cnv_df.reset_index(), on = "submitter_id").set_index("submitter_id")
    tcga_train_merged_df = pd.merge(tcga_train_merged_mut_cnv_df.reset_index(), tcga_train_response_drug_df.drop("drug_name", axis = 1), on = "submitter_id").set_index("submitter_id")

#     tcga_train_merged_df = pd.merge(tcga_train_mut_df.reset_index(), tcga_train_response_drug_df.drop("drug_name", axis = 1), on = "submitter_id").set_index("submitter_id")
    if is_train and (len(tcga_train_merged_df) > 1) and not train_all:
#         print(tcga_train_merged_df.shape)
        df_train, df_val = train_test_split(tcga_train_merged_df, test_size=0.2, random_state=42, stratify=tcga_train_merged_df['response'])
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(tcga_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for TCGA {type_set} set')
    return tcga_train_merged_df

def get_dataset_benchmark(domain_name, drug_name, input_name, type_set=''):
    if input_name == 'mutation':
        if domain_name == 'pdx':
            return get_pdx(drug_name, type_set)
        elif domain_name == 'tcga':
            return get_tcga(drug_name, type_set)
        else:
            raise ValueError(
                        f"Unsupported domain - {domain_name} - accepted types are [pdx, tcga] "
                    )  
    elif input_name == 'cnv':
        if domain_name == 'pdx':
            return get_pdx_cnv(drug_name, type_set)
        elif domain_name == 'tcga':
            return get_tcga_cnv(drug_name, type_set)
        else:
            raise ValueError(
                        f"Unsupported domain - {domain_name} - accepted types are [pdx, tcga] "
                    ) 
    elif input_name == 'mut_cnv':
        if domain_name == 'pdx':
            return get_pdx_mut_cnv(drug_name, type_set)
        elif domain_name == 'tcga':
            return get_tcga_mut_cnv(drug_name, type_set)
        else:
            raise ValueError(
                        f"Unsupported domain - {domain_name} - accepted types are [pdx, tcga] "
                    ) 
    else:
        raise ValueError(
            f"Unsupported input - {input_name} - accepted types are [mutation, cnv, mut_cnv] "
        ) 