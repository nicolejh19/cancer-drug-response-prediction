import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from dataset import (
    AggCategoricalAnnotatedCellLineDataset,
    AggCategoricalAnnotatedPdxDataset,
    AggCategoricalAnnotatedTcgaDataset    
)

def get_cl_threshold(drug_name):
    random.seed(42)
    np.random.seed(42)
    
    cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
        is_train=True,
        only_cat_one_drugs=True,
        scale_y=False,
        use_k_best_worst=None,
    )
    cl_train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    cl_train_mut_df = cl_dataset_train.raw_mutations[cl_dataset_train.raw_mutations.index.isin(cl_train_y_drug_df.depmap_id)]
    cl_train_merged_df = pd.merge(cl_train_mut_df.reset_index(), cl_train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")
    
    df_train, df_val = train_test_split(cl_train_merged_df, test_size=0.2, random_state=42)
#     print(np.quantile(df_train.auc, [0,0.25,0.5,0.75,1]))
    threshold = np.round(np.percentile(df_train.auc, 25), 6)
    if threshold >= 0.5:
        print(f'\nUsing 10th percentile AUDRC as threshold for cell-line with {drug_name}')
        threshold = np.round(np.percentile(df_train.auc, 10), 6)
    else:
        print(f'\nUsing 25th percentile AUDRC as threshold for cell-line with {drug_name}')
    return threshold

def get_cl(drug_name, type_set, flags):
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
    cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
        is_train=is_train,
        only_cat_one_drugs=True,
        scale_y=False,
        use_k_best_worst=None,
    )
    cl_train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    cl_train_mut_df = cl_dataset_train.raw_mutations[cl_dataset_train.raw_mutations.index.isin(cl_train_y_drug_df.depmap_id)]
    cl_train_merged_df = pd.merge(cl_train_mut_df.reset_index(), cl_train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")
    
#     threshold = get_cl_threshold(drug_name)
    threshold = flags['audrc_threshold']
    print(f'Threshold used for cell-line with {drug_name}: {threshold}')
    
    cl_train_merged_df['response'] = cl_train_merged_df['auc'] <= threshold
    cl_train_merged_df['response'] = cl_train_merged_df['response'].astype(int)
    cl_train_merged_df = cl_train_merged_df.drop(columns="auc")

    if is_train and (len(cl_train_merged_df) > 1) and not train_all:
        df_train, df_val = train_test_split(cl_train_merged_df, test_size=0.2, random_state=42)
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(cl_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for cell-line {type_set} set')
    return cl_train_merged_df
    
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

    if is_train and (len(tcga_train_merged_df) > 1) and not train_all:
        df_train, df_val = train_test_split(tcga_train_merged_df, test_size=0.2, random_state=42, stratify=tcga_train_merged_df['response'])
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(tcga_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for TCGA {type_set} set')
    return tcga_train_merged_df

def get_cl_threshold_cnv(drug_name):
    random.seed(42)
    np.random.seed(42)
    
    cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
        is_train=True,
        only_cat_one_drugs=True,
        scale_y=False,
        use_k_best_worst=None,
    )
    cl_train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    cl_train_cnv_df = cl_dataset_train.cnv[cl_dataset_train.cnv.index.isin(cl_train_y_drug_df.depmap_id)]
    cl_train_merged_df = pd.merge(cl_train_cnv_df.reset_index(), cl_train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")
    
    df_train, df_val = train_test_split(cl_train_merged_df, test_size=0.2, random_state=42)
#     print(np.quantile(df_train.auc, [0,0.25,0.5,0.75,1]))
    threshold = np.round(np.percentile(df_train.auc, 25), 6)
    if threshold >= 0.5:
        print(f'\nUsing 10th percentile AUDRC as threshold for cell-line with {drug_name}')
        threshold = np.round(np.percentile(df_train.auc, 10), 6)
    else:
        print(f'\nUsing 25th percentile AUDRC as threshold for cell-line with {drug_name}')
    return threshold

def get_cl_cnv(drug_name, type_set, flags):
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
    cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
        is_train=is_train,
        only_cat_one_drugs=True,
        scale_y=False,
        use_k_best_worst=None,
    )
    cl_train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    cl_train_cnv_df = cl_dataset_train.cnv[cl_dataset_train.cnv.index.isin(cl_train_y_drug_df.depmap_id)]
    cl_train_merged_df = pd.merge(cl_train_cnv_df.reset_index(), cl_train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")
    
#     threshold = get_cl_threshold_cnv(drug_name)
    threshold = flags['audrc_threshold']
    print(f'Threshold used for cell-line with {drug_name}: {threshold}')
    
    cl_train_merged_df['response'] = cl_train_merged_df['auc'] <= threshold
    cl_train_merged_df['response'] = cl_train_merged_df['response'].astype(int)
    cl_train_merged_df = cl_train_merged_df.drop(columns="auc")

    if is_train and (len(cl_train_merged_df) > 1) and not train_all:
        df_train, df_val = train_test_split(cl_train_merged_df, test_size=0.2, random_state=42)
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(cl_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for cell-line {type_set} set')
    return cl_train_merged_df

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

def get_cl_threshold_mut_cnv(drug_name):
    random.seed(42)
    np.random.seed(42)
    
    cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
        is_train=True,
        only_cat_one_drugs=True,
        scale_y=False,
        use_k_best_worst=None,
    )
    cl_train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    cl_train_mut_df = cl_dataset_train.raw_mutations[cl_dataset_train.raw_mutations.index.isin(cl_train_y_drug_df.depmap_id)]
    cl_train_cnv_df = cl_dataset_train.cnv[cl_dataset_train.cnv.index.isin(cl_train_y_drug_df.depmap_id)]
    cl_train_merged_mut_cnv_df = pd.merge(cl_train_mut_df.reset_index(), cl_train_cnv_df.reset_index(), on = "depmap_id").set_index("depmap_id")
    cl_train_merged_df = pd.merge(cl_train_merged_mut_cnv_df.reset_index(), cl_train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")

    df_train, df_val = train_test_split(cl_train_merged_df, test_size=0.2, random_state=42)
#     print(np.quantile(df_train.auc, [0,0.25,0.5,0.75,1]))
    threshold = np.round(np.percentile(df_train.auc, 25), 6)
    if threshold >= 0.5:
        print(f'\nUsing 10th percentile AUDRC as threshold for cell-line with {drug_name}')
        threshold = np.round(np.percentile(df_train.auc, 10), 6)
    else:
        print(f'\nUsing 25th percentile AUDRC as threshold for cell-line with {drug_name}')
    return threshold

def get_cl_mut_cnv(drug_name, type_set, flags):
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
    cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
        is_train=is_train,
        only_cat_one_drugs=True,
        scale_y=False,
        use_k_best_worst=None,
    )
    cl_train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
    cl_train_mut_df = cl_dataset_train.raw_mutations[cl_dataset_train.raw_mutations.index.isin(cl_train_y_drug_df.depmap_id)]
    
    cl_train_cnv_df = cl_dataset_train.cnv[cl_dataset_train.cnv.index.isin(cl_train_y_drug_df.depmap_id)]
    cl_train_merged_mut_cnv_df = pd.merge(cl_train_mut_df.reset_index(), cl_train_cnv_df.reset_index(), on = "depmap_id").set_index("depmap_id")
    cl_train_merged_df = pd.merge(cl_train_merged_mut_cnv_df.reset_index(), cl_train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")

#     threshold = get_cl_threshold_mut_cnv(drug_name)
    threshold = flags['audrc_threshold']
    print(f'Threshold used for cell-line with {drug_name}: {threshold}')
    
    cl_train_merged_df['response'] = cl_train_merged_df['auc'] <= threshold
    cl_train_merged_df['response'] = cl_train_merged_df['response'].astype(int)
    cl_train_merged_df = cl_train_merged_df.drop(columns="auc")

    if is_train and (len(cl_train_merged_df) > 1) and not train_all:
#         print(cl_train_merged_df.shape)
        df_train, df_val = train_test_split(cl_train_merged_df, test_size=0.2, random_state=42)
        if type_set == 'train':
            return df_train
        else:
            return df_val
    if len(cl_train_merged_df) == 1:
        print(f'{drug_name} has only 1 samples for cell-line {type_set} set')
    return cl_train_merged_df

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

def get_dataset(domain_name, drug_name, type_set, flags, input_name):
    if input_name == 'mutation':
        if domain_name == 'cl':
            return get_cl(drug_name, type_set, flags)
        elif domain_name == 'pdx':
            return get_pdx(drug_name, type_set)
        elif domain_name == 'tcga':
            return get_tcga(drug_name, type_set)
        else:
            raise ValueError(
                        f"Unsupported domain - {domain_name} - accepted types are [cl, pdx, tcga] "
                    )
    elif input_name == 'cnv':
        if domain_name == 'cl':
            return get_cl_cnv(drug_name, type_set, flags)
        elif domain_name == 'pdx':
            return get_pdx_cnv(drug_name, type_set)
        elif domain_name == 'tcga':
            return get_tcga_cnv(drug_name, type_set)
        else:
            raise ValueError(
                        f"Unsupported domain - {domain_name} - accepted types are [cl, pdx, tcga] "
                    )  
    elif input_name == 'mut_cnv':
        if domain_name == 'cl':
            return get_cl_mut_cnv(drug_name, type_set, flags)
        elif domain_name == 'pdx':
            return get_pdx_mut_cnv(drug_name, type_set)
        elif domain_name == 'tcga':
            return get_tcga_mut_cnv(drug_name, type_set)
        else:
            raise ValueError(
                        f"Unsupported domain - {domain_name} - accepted types are [cl, pdx, tcga] "
                    ) 
    else:
        raise ValueError(
            f"Unsupported input - {input_name} - accepted types are [mutations, cnv, mut_cnv] "
        ) 