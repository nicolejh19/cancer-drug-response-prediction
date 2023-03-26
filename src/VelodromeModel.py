import pandas as pd
import numpy as np
import os

import torch
import random
from torch import nn
from torch.nn import functional as F

from functools import cached_property

from torch.nn import Linear, ReLU, Sequential
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split

from dataset import (
    AggCategoricalAnnotatedCellLineDataset,
    AggCategoricalAnnotatedPdxDataset,
    AggCategoricalAnnotatedTcgaDataset    
)

from collections import Counter
import seaborn as sns

from AdaptFuncVelov3 import *
from NetVelo import *

import ax
from ax import RangeParameter, ChoiceParameter, FixedParameter
from ax import ParameterType, SearchSpace
from ax.service.managed_loop import optimize

# instantiate 1 object for each drug 
class VelodromeMainMut(nn.Module):
    def __init__(self, drug_name):
        super(VelodromeMainMut, self).__init__()
        self.drug = drug_name
        self.save_models = f"./velodrome_models/replicate_mutations/tuned/{drug_name}"
        if not os.path.isdir(self.save_models):
            os.makedirs(self.save_models)
        self.save_results = f"./velodrome_results/replicate_mutations/tuned/{drug_name}"
        if not os.path.isdir(self.save_results):
            os.makedirs(self.save_results)
        self.prediction_path = f'./velodrome_predictions/replicate_mutations/tuned/{drug_name}'
        if not os.path.isdir(self.prediction_path):
            os.makedirs(self.prediction_path)
    
    def setup(self, args):
        args['lam2'] = 1 - args['lam1']
        self.args = args
        torch.manual_seed(args['seed'])
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        
        # load cell line train data
        cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
            is_train=True,
            only_cat_one_drugs=True,
            scale_y=False,
            use_k_best_worst=None,
        )

        # unlabelled TCGA dataset
        tcga_train = AggCategoricalAnnotatedTcgaDataset(
            is_train=True,
            only_cat_one_drugs=True,
        )
        # filter out for the specific drug (train)
        train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == self.drug] # has depmap_id, drug_name, auc 
        train_mut_df = cl_dataset_train.raw_mutations[cl_dataset_train.raw_mutations.index.isin(train_y_drug_df.depmap_id)]
        train_merged_df = pd.merge(train_mut_df.reset_index(), train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")

        df_train, df_val = train_test_split(train_merged_df, test_size=0.2, random_state=42)

        # divide into 2 cell line datasets to be passed into the 2 predictors
        X1_train = df_train.iloc[0: len(df_train)//2].drop("auc", axis = 1).values
        y1_train = df_train.iloc[0: len(df_train)//2]["auc"].values
        X2_train = df_train.iloc[len(df_train)//2:].drop("auc", axis = 1).values
        y2_train = df_train.iloc[len(df_train)//2:]["auc"].values
        self.X1_train = X1_train 

        train1Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X1_train), torch.FloatTensor(y1_train))
        self.trainLoader_1 = torch.utils.data.DataLoader(dataset = train1Dataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        train2Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X2_train), torch.FloatTensor(y2_train))
        self.trainLoader_2 = torch.utils.data.DataLoader(dataset = train2Dataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        X_val = df_val.drop("auc", axis = 1).values
        y_val = df_val.iloc[:]["auc"].values
        self.TX_val = torch.FloatTensor(X_val)
        self.Ty_val = torch.FloatTensor(y_val)

#             tcga_train_response_drug_df = tcga_train.tcga_response[tcga_train.tcga_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
        tcga_train_response_drug_df = tcga_train.tcga_response # considering all train TCGA entities (if filter for specific drug (docetaxel) get empty tensor for loss1)
        tcga_train_mut_df = tcga_train.raw_mutations[tcga_train.raw_mutations.index.isin(tcga_train_response_drug_df.submitter_id)]
#             tcga_train_merged_df = pd.merge(tcga_train_mut_df.reset_index(), tcga_train_response_drug_df.drop("drug_name", axis = 1), on = "submitter_id").set_index("submitter_id")
        X_U = tcga_train_mut_df.values

        trainUDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_U))
        self.trainULoader = torch.utils.data.DataLoader(dataset = trainUDataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        print('\n--------------------', self.drug, '--------------------')
        print(f'Cell-line dataset for training Predictor 1: {X1_train.shape}')
        print(f'Cell-line dataset for training Predictor 2: {X2_train.shape}')
        print(f'Total no. of cell-lines for training: {df_train.shape[0]}')
        print(f'TCGA dataset for training both Predictors: {X_U.shape}')
        print(f'Total no. of cell-lines for evaluation: {df_val.shape[0]}')
            
        # instantiate network
        self.model, self.pred1, self.pred2 = Network(args=self.args, X=self.X1_train)
        self.w1 = [torch.tensor(0.5)] * args['epoch']
        self.w2 = [torch.tensor(0.5)] * args['epoch']
        
    def train(self, args): # follows Velodrome.py 
#         self.model, self.pred1, self.pred2 = Network(args=self.args, X=self.X1_train)
        self.setup(args)
        # optimisers
        opt = torch.optim.Adagrad(self.model.parameters(), lr=self.args['lr'], weight_decay = self.args['wd'])
        opt1 = torch.optim.Adagrad(self.pred1.parameters(), lr=self.args['lr1'], weight_decay = self.args['wd1'])
        opt2 = torch.optim.Adagrad(self.pred2.parameters(), lr=self.args['lr2'], weight_decay = self.args['wd2'])
        
        best_pr = -np.inf # allow for negative pearson correlation coefficient
        
        loss_fun = torch.nn.MSELoss()
        total_val = []
        total_aac = []

        train_loss = []
        consistency_loss = []
        covariance_loss = []
        train_pr1 = []
        train_pr2 = []
        val_loss = []
        val_pr = []
    
        train_pred = []
        w1 = []
        w2 = []
        
        for ite in range(self.args['epoch']):
            torch.autograd.set_detect_anomaly(True)
            pred_loss, coral_loss, con_loss, epoch_pr1, epoch_pr2, loss1, loss2 = train(self.args, self.model, self.pred1, self.pred2, loss_fun, opt, opt1, opt2, self.trainLoader_1, self.trainLoader_2, self.trainULoader)

            train_loss.append(pred_loss + coral_loss + con_loss)      
            train_loss.append(pred_loss + con_loss)      
            consistency_loss.append(con_loss)
            covariance_loss.append(coral_loss)
            train_pr1.append(epoch_pr1)
            train_pr2.append(epoch_pr2)

            w1.append(loss1)
            w2.append(loss2)

            epoch_val_loss, epoch_Val_pr,_ = validate_workflow(self.args, self.model, self.pred1, self.pred2, [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))], loss_fun, self.TX_val, self.Ty_val)
            val_loss.append(epoch_val_loss)
            val_pr.append(epoch_Val_pr)                      
#             print(epoch_Val_pr)
            if epoch_Val_pr > best_pr: 
                best_pr = epoch_Val_pr
                torch.save(self.model.state_dict(), os.path.join(self.save_models, 'Best_Model.pt'))
                torch.save(self.pred1.state_dict(), os.path.join(self.save_models, 'Best_Pred1.pt'))
                torch.save(self.pred2.state_dict(), os.path.join(self.save_models, 'Best_Pred2.pt'))

#         plots(self.args, train_loss, consistency_loss, covariance_loss, train_pr1, train_pr2, val_loss, val_pr)
        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))

        self.model.eval()
        self.pred1.eval()
        self.pred2.eval()

        _,_, preds= validate_workflow(self.args, self.model, self.pred1, self.pred2, [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))], loss_fun, self.TX_val, self.Ty_val)
        total_val.append(preds.detach().numpy().flatten())
        total_aac.append(self.Ty_val.detach().numpy())
        self.w1 = w1
        self.w2 = w2

        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))


        final_pred = list(itertools.chain.from_iterable(total_val))
        final_labels = list(itertools.chain.from_iterable(total_aac))  
        dct = {'pr': best_pr}
        print(f'Highest Pearson r achieved during training: {best_pr}')
        return dct
                                   
    def forward(self, X): # follows FuncVelov3's validate_workflow
        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))
        
        self.model.eval()
        self.pred1.eval()
        self.pred2.eval()
        
        ws = [torch.mean(torch.stack(self.w1)), torch.mean(torch.stack(self.w2))]
        w_n = torch.nn.functional.softmax(torch.stack(ws), dim=None)
        w1 = w_n[0]
        w2 = w_n[1]
        
        TX_val = torch.tensor(
            X,
            dtype=torch.float,
        )
        fx_val = self.model(TX_val)
        pred_1 = self.pred1(fx_val)
        pred_2 = self.pred2(fx_val)
        pred_val = w1*pred_1+w2*pred_2 # weighted average
        return pred_val
        
# instantiate for running test bed evaluation
class VelodromeTestBedMut():
    def __init__(self, drug_name):
        super(VelodromeTestBedMut, self).__init__()
        self.main_model = VelodromeMainMut(drug_name)
        self.drug_name = drug_name
        
    def train_model(self, best_parameters):
        self.main_model.train(best_parameters)
            
    def tune_model(self, parameters, total_trials):
        best_parameters, values, experiment, model = optimize(
            experiment_name="velodrome_replicate",
            parameters=parameters,
            evaluation_function=self.main_model.train,
            objective_name='pr',
            minimize=False,
            total_trials=total_trials
        )
        return best_parameters, values, experiment, model
                                   
    def get_velodrome_results_df(self, dataset, dataset_name='', save_predictions=False):
        X = dataset.raw_mutations
        unique_entities = X.index.unique()
        y_all = pd.concat(list(dataset[: len(dataset)].values()), axis = 1)
        drug_name = self.drug_name
        
        y = y_all[y_all.drug_name == drug_name]
        y_pred = []

        if y.shape[0] == 0:
            print(f"WARNING: {drug_name} does not exist in dataset")
            return pd.DataFrame()

        for idx, row in y.iterrows():
            y_pred.append(self.main_model(torch.tensor(X.loc[row[dataset.entity_identifier_name]]).view(1, -1)))
        np_out = torch.tensor(y_pred).cpu().detach().numpy()

        y_true = y.copy()
        y_pred = y_true.copy()

        if isinstance(dataset, (AggCategoricalAnnotatedPdxDataset,AggCategoricalAnnotatedTcgaDataset)):
            y_pred["response"] = 1 - np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "response"]
            ].copy()
            measure_column_name = "response"
            if len(np.unique(y_true[measure_column_name])) == 1:
                print(f"Dataset has only 1 class label for {drug_name}")
                drug_roc_auc = np.nan
            else:
                drug_roc_auc = roc_auc_score(y_true[measure_column_name], y_pred[measure_column_name])
            drug_spearmanr = spearmanr(y_true[measure_column_name], y_pred[measure_column_name])
            if len(y_true) < 2:
                drug_pearsonr = (np.nan, np.nan)
            else:
                drug_pearsonr = pearsonr(y_true[measure_column_name], y_pred[measure_column_name])

            drug_results_df = pd.DataFrame(data={'drug_name': [drug_name],
                                                 'length_dataset': [len(y_true)],
                                                'roc_auc': [drug_roc_auc], 
                                                'aupr': [average_precision_score(y_true[measure_column_name], y_pred[measure_column_name])], 
                                                'spearmanr_correlation': [drug_spearmanr[0]], 
                                                'spearmanr_pvalue': [drug_spearmanr[1]], 
                                                'pearsonr_correlation': [drug_pearsonr[0]],
                                                'pearsonr_pvalue': [drug_pearsonr[1]]})

        elif isinstance(dataset, AggCategoricalAnnotatedCellLineDataset):
            y_pred["auc"] = np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "auc"]
            ].copy()
            measure_column_name = "auc"
            drug_spearmanr = spearmanr(y_true[measure_column_name], y_pred[measure_column_name])
            drug_pearsonr = pearsonr(y_true[measure_column_name], y_pred[measure_column_name])
            drug_results_df = pd.DataFrame(data={'drug_name': [drug_name],
                                                 'length_dataset': [len(y_true)],
                                                'spearmanr_correlation': [drug_spearmanr[0]], 
                                                'spearmanr_pvalue': [drug_spearmanr[1]], 
                                                'pearsonr_correlation': [drug_pearsonr[0]],
                                                'pearsonr_pvalue': [drug_pearsonr[1]]})
        else:
            raise ValueError(
                f"Unsupported dataset type - {type(dataset)} - accepted types are [AggCategoricalAnnotatedCellLineDataset, AggCategoricalAnnotatedPdxDataset, AggCategoricalAnnotatedTcgaDataset]"
            )    
            
        if save_predictions:
            np.savetxt(os.path.join(self.main_model.prediction_path, f'{dataset_name}_testset_groundtruth.csv'), y_true[measure_column_name])
            np.savetxt(os.path.join(self.main_model.prediction_path, f'{dataset_name}_testset_predictions.csv'), y_pred[measure_column_name])

        return drug_results_df
    
    def get_roc_pr_curves(self, drug_name, dataset):
        X = dataset.raw_mutations
        unique_entities = X.index.unique()
        y_all = pd.concat(list(dataset[: len(dataset)].values()), axis = 1)

        y = y_all[y_all.drug_name == drug_name]
        y_pred = []

        if y.shape[0] == 0:
            print(f"WARNING: {drug_name} does not exist in dataset")
            return

        for idx, row in y.iterrows():
            y_pred.append(self.main_model(torch.tensor(X.loc[row[dataset.entity_identifier_name]]).view(1, -1)))
        np_out = torch.tensor(y_pred).cpu().detach().numpy()

        y_true = y.copy()
        y_pred = y_true.copy()

        if isinstance(dataset, (AggCategoricalAnnotatedPdxDataset,AggCategoricalAnnotatedTcgaDataset)):
            y_pred["response"] = 1 - np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "response"]
            ].copy()
            measure_column_name = "response"
            if len(np.unique(y_true[measure_column_name])) == 1:
                print(f"Dataset has only 1 class label for {drug_name}")
                return

            drug_roc_auc = roc_auc_score(y_true[measure_column_name], y_pred[measure_column_name])
            aupr = average_precision_score(y_true[measure_column_name], y_pred[measure_column_name])
            fpr, tpr, thresholds = roc_curve(y_true[measure_column_name], y_pred[measure_column_name])
            # generate no skill predictions (majority class)
            c = Counter(y_true[measure_column_name])
            ns_probs = [c.most_common(1)[0][0] for _ in range(len(y_true))]
            ns_auc = roc_auc_score(y_true[measure_column_name], ns_probs)
            ns_fpr, ns_tpr, _ = roc_curve(y_true[measure_column_name], ns_probs)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].plot(fpr, tpr, color="navy", lw= 2, label="Model (area = %0.4f)" % drug_roc_auc)
            axs[0].plot(ns_fpr, ns_tpr, color="grey", linestyle='--', label='No Skill')
            axs[0].set_xlim([0.0, 1.0])
            axs[0].set_ylim([0.0, 1.05])
            axs[0].set_xlabel("False Positive Rate")
            axs[0].set_ylabel("True Positive Rate")
            axs[0].set_title("Receiver Operating Characteristic (ROC) curve")
            axs[0].legend(loc="lower right")

            precision, recall, thresholds = precision_recall_curve(y_true[measure_column_name], y_pred[measure_column_name])
    #         ns_precision, ns_recall, _ = precision_recall_curve(y_true[measure_column_name], ns_probs)
            axs[1].plot(recall, precision, color='purple', lw= 2, label="Model (area = %0.4f)" % aupr)
            axs[1].set_title('Precision-Recall Curve')
            axs[1].set_xlim([0.0, 1.0])
            axs[1].set_ylim([0.0, 1.05])
            axs[1].set_ylabel('Precision')
            axs[1].set_xlabel('Recall')
            no_skill = len(y_true[y_true[measure_column_name]==1]) / len(y_true)
            axs[1].plot([0, 1], [no_skill, no_skill], color="grey", linestyle='--', label='No Skill')
    #         axs[1].plot(ns_recall, ns_precision, color="grey", linestyle='--', label='No Skill')
            axs[1].legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(
                f"Unsupported dataset type - {type(dataset)} - accepted types are [AggCategoricalAnnotatedPdxDataset, AggCategoricalAnnotatedTcgaDataset]"
            )    
            
# instantiate 1 object for each drug 
class VelodromeMainCNV(nn.Module):
    def __init__(self, drug_name):
        super(VelodromeMainCNV, self).__init__()
        self.drug = drug_name
        self.save_models = f"./velodrome_models/replicate_cnv/tuned/{drug_name}"
        if not os.path.isdir(self.save_models):
            os.makedirs(self.save_models)
        self.save_results = f"./velodrome_results/replicate_cnv/tuned/{drug_name}"
        if not os.path.isdir(self.save_results):
            os.makedirs(self.save_results)
        self.prediction_path = f'./velodrome_predictions/replicate_cnv/tuned/{drug_name}'
        if not os.path.isdir(self.prediction_path):
            os.makedirs(self.prediction_path)
    
    def setup(self, args):
        args['lam2'] = 1 - args['lam1']
        self.args = args
        torch.manual_seed(args['seed'])
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        
        # load cell line train data
        cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
            is_train=True,
            only_cat_one_drugs=True,
            scale_y=False,
            use_k_best_worst=None,
        )

        # unlabelled TCGA dataset
        tcga_train = AggCategoricalAnnotatedTcgaDataset(
            is_train=True,
            only_cat_one_drugs=True,
        )
        # filter out for the specific drug (train)
        train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == self.drug] # has depmap_id, drug_name, auc 
        train_cnv_df = cl_dataset_train.cnv[cl_dataset_train.cnv.index.isin(train_y_drug_df.depmap_id)]
        train_merged_df = pd.merge(train_cnv_df.reset_index(), train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")

        df_train, df_val = train_test_split(train_merged_df, test_size=0.2, random_state=42)

        # divide into 2 cell line datasets to be passed into the 2 predictors
        X1_train = df_train.iloc[0: len(df_train)//2].drop("auc", axis = 1).values
        y1_train = df_train.iloc[0: len(df_train)//2]["auc"].values
        X2_train = df_train.iloc[len(df_train)//2:].drop("auc", axis = 1).values
        y2_train = df_train.iloc[len(df_train)//2:]["auc"].values
        self.X1_train = X1_train 

        train1Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X1_train), torch.FloatTensor(y1_train))
        self.trainLoader_1 = torch.utils.data.DataLoader(dataset = train1Dataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        train2Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X2_train), torch.FloatTensor(y2_train))
        self.trainLoader_2 = torch.utils.data.DataLoader(dataset = train2Dataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        X_val = df_val.drop("auc", axis = 1).values
        y_val = df_val.iloc[:]["auc"].values
        self.TX_val = torch.FloatTensor(X_val)
        self.Ty_val = torch.FloatTensor(y_val)

#             tcga_train_response_drug_df = tcga_train.tcga_response[tcga_train.tcga_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
        tcga_train_response_drug_df = tcga_train.tcga_response # considering all train TCGA entities (if filter for specific drug (docetaxel) get empty tensor for loss1)
        tcga_train_cnv_df = tcga_train.cnv[tcga_train.cnv.index.isin(tcga_train_response_drug_df.submitter_id)]
#             tcga_train_merged_df = pd.merge(tcga_train_cnv_df.reset_index(), tcga_train_response_drug_df.drop("drug_name", axis = 1), on = "submitter_id").set_index("submitter_id")
        X_U = tcga_train_cnv_df.values

        trainUDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_U))
        self.trainULoader = torch.utils.data.DataLoader(dataset = trainUDataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        print('\n--------------------', self.drug, '--------------------')
        print(f'Cell-line dataset for training Predictor 1: {X1_train.shape}')
        print(f'Cell-line dataset for training Predictor 2: {X2_train.shape}')
        print(f'Total no. of cell-lines for training: {df_train.shape[0]}')
        print(f'TCGA dataset for training both Predictors: {X_U.shape}')
        print(f'Total no. of cell-lines for evaluation: {df_val.shape[0]}')
            
        # instantiate network
        self.model, self.pred1, self.pred2 = Network(args=self.args, X=self.X1_train)
        self.w1 = [torch.tensor(0.5)] * args['epoch']
        self.w2 = [torch.tensor(0.5)] * args['epoch']
        
    def train(self, args): # follows Velodrome.py 
#         self.model, self.pred1, self.pred2 = Network(args=self.args, X=self.X1_train)
        self.setup(args)
        # optimisers
        opt = torch.optim.Adagrad(self.model.parameters(), lr=self.args['lr'], weight_decay = self.args['wd'])
        opt1 = torch.optim.Adagrad(self.pred1.parameters(), lr=self.args['lr1'], weight_decay = self.args['wd1'])
        opt2 = torch.optim.Adagrad(self.pred2.parameters(), lr=self.args['lr2'], weight_decay = self.args['wd2'])
        
        best_pr = -np.inf # allow for negative pearson correlation coefficient
        
        loss_fun = torch.nn.MSELoss()
        total_val = []
        total_aac = []

        train_loss = []
        consistency_loss = []
        covariance_loss = []
        train_pr1 = []
        train_pr2 = []
        val_loss = []
        val_pr = []
    
        train_pred = []
        w1 = []
        w2 = []
        
        for ite in range(self.args['epoch']):
            torch.autograd.set_detect_anomaly(True)
            pred_loss, coral_loss, con_loss, epoch_pr1, epoch_pr2, loss1, loss2 = train(self.args, self.model, self.pred1, self.pred2, loss_fun, opt, opt1, opt2, self.trainLoader_1, self.trainLoader_2, self.trainULoader)

            train_loss.append(pred_loss + coral_loss + con_loss)      
            train_loss.append(pred_loss + con_loss)      
            consistency_loss.append(con_loss)
            covariance_loss.append(coral_loss)
            train_pr1.append(epoch_pr1)
            train_pr2.append(epoch_pr2)

            w1.append(loss1)
            w2.append(loss2)

            epoch_val_loss, epoch_Val_pr,_ = validate_workflow(self.args, self.model, self.pred1, self.pred2, [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))], loss_fun, self.TX_val, self.Ty_val)
            val_loss.append(epoch_val_loss)
            val_pr.append(epoch_Val_pr)                      
#             print(epoch_Val_pr)
            if epoch_Val_pr > best_pr: 
                best_pr = epoch_Val_pr
                torch.save(self.model.state_dict(), os.path.join(self.save_models, 'Best_Model.pt'))
                torch.save(self.pred1.state_dict(), os.path.join(self.save_models, 'Best_Pred1.pt'))
                torch.save(self.pred2.state_dict(), os.path.join(self.save_models, 'Best_Pred2.pt'))

#         plots(self.args, train_loss, consistency_loss, covariance_loss, train_pr1, train_pr2, val_loss, val_pr)
        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))

        self.model.eval()
        self.pred1.eval()
        self.pred2.eval()

        _,_, preds= validate_workflow(self.args, self.model, self.pred1, self.pred2, [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))], loss_fun, self.TX_val, self.Ty_val)
        total_val.append(preds.detach().numpy().flatten())
        total_aac.append(self.Ty_val.detach().numpy())
        self.w1 = w1
        self.w2 = w2

        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))


        final_pred = list(itertools.chain.from_iterable(total_val))
        final_labels = list(itertools.chain.from_iterable(total_aac))  
        dct = {'pr': best_pr}
        print(f'Highest Pearson r achieved during training: {best_pr}')
        return dct
                                   
    def forward(self, X): # follows FuncVelov3's validate_workflow
        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))
        
        self.model.eval()
        self.pred1.eval()
        self.pred2.eval()
        
        ws = [torch.mean(torch.stack(self.w1)), torch.mean(torch.stack(self.w2))]
        w_n = torch.nn.functional.softmax(torch.stack(ws), dim=None)
        w1 = w_n[0]
        w2 = w_n[1]
        
        TX_val = torch.tensor(
            X,
            dtype=torch.float,
        )
        fx_val = self.model(TX_val)
        pred_1 = self.pred1(fx_val)
        pred_2 = self.pred2(fx_val)
        pred_val = w1*pred_1+w2*pred_2 # weighted average
        return pred_val

# instantiate for running test bed evaluation
class VelodromeTestBedCNV():
    def __init__(self, drug_name):
        super(VelodromeTestBedCNV, self).__init__()
        self.main_model = VelodromeMainCNV(drug_name)
        self.drug_name = drug_name
        
    def train_model(self, best_parameters):
        self.main_model.train(best_parameters)
            
    def tune_model(self, parameters, total_trials):
        best_parameters, values, experiment, model = optimize(
            experiment_name="velodrome_replicate",
            parameters=parameters,
            evaluation_function=self.main_model.train,
            objective_name='pr',
            minimize=False,
            total_trials=total_trials
        )
        return best_parameters, values, experiment, model
                                   
    def get_velodrome_results_df(self, dataset, dataset_name='', save_predictions=False):
        X = dataset.cnv
        unique_entities = X.index.unique()
        y_all = pd.concat(list(dataset[: len(dataset)].values()), axis = 1)
        drug_name = self.drug_name
        
        y = y_all[y_all.drug_name == drug_name]
        y_pred = []

        if y.shape[0] == 0:
            print(f"WARNING: {drug_name} does not exist in dataset")
            return pd.DataFrame()

        for idx, row in y.iterrows():
            y_pred.append(self.main_model(torch.tensor(X.loc[row[dataset.entity_identifier_name]]).view(1, -1)))
        np_out = torch.tensor(y_pred).cpu().detach().numpy()

        y_true = y.copy()
        y_pred = y_true.copy()

        if isinstance(dataset, (AggCategoricalAnnotatedPdxDataset,AggCategoricalAnnotatedTcgaDataset)):
            y_pred["response"] = 1 - np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "response"]
            ].copy()
            measure_column_name = "response"
            if len(np.unique(y_true[measure_column_name])) == 1:
                print(f"Dataset has only 1 class label for {drug_name}")
                drug_roc_auc = np.nan
            else:
                drug_roc_auc = roc_auc_score(y_true[measure_column_name], y_pred[measure_column_name])
            drug_spearmanr = spearmanr(y_true[measure_column_name], y_pred[measure_column_name])
            if len(y_true) < 2:
                drug_pearsonr = (np.nan, np.nan)
            else:
                drug_pearsonr = pearsonr(y_true[measure_column_name], y_pred[measure_column_name])

            drug_results_df = pd.DataFrame(data={'drug_name': [drug_name],
                                                 'length_dataset': [len(y_true)],
                                                'roc_auc': [drug_roc_auc], 
                                                'aupr': [average_precision_score(y_true[measure_column_name], y_pred[measure_column_name])], 
                                                'spearmanr_correlation': [drug_spearmanr[0]], 
                                                'spearmanr_pvalue': [drug_spearmanr[1]], 
                                                'pearsonr_correlation': [drug_pearsonr[0]],
                                                'pearsonr_pvalue': [drug_pearsonr[1]]})

        elif isinstance(dataset, AggCategoricalAnnotatedCellLineDataset):
            y_pred["auc"] = np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "auc"]
            ].copy()
            measure_column_name = "auc"
            drug_spearmanr = spearmanr(y_true[measure_column_name], y_pred[measure_column_name])
            drug_pearsonr = pearsonr(y_true[measure_column_name], y_pred[measure_column_name])
            drug_results_df = pd.DataFrame(data={'drug_name': [drug_name],
                                                 'length_dataset': [len(y_true)],
                                                'spearmanr_correlation': [drug_spearmanr[0]], 
                                                'spearmanr_pvalue': [drug_spearmanr[1]], 
                                                'pearsonr_correlation': [drug_pearsonr[0]],
                                                'pearsonr_pvalue': [drug_pearsonr[1]]})
        else:
            raise ValueError(
                f"Unsupported dataset type - {type(dataset)} - accepted types are [AggCategoricalAnnotatedCellLineDataset, AggCategoricalAnnotatedPdxDataset, AggCategoricalAnnotatedTcgaDataset]"
            )    
            
        if save_predictions:
            np.savetxt(os.path.join(self.main_model.prediction_path, f'{dataset_name}_testset_groundtruth.csv'), y_true[measure_column_name])
            np.savetxt(os.path.join(self.main_model.prediction_path, f'{dataset_name}_testset_predictions.csv'), y_pred[measure_column_name])

        return drug_results_df
    
    def get_roc_pr_curves(self, drug_name, dataset):
        X = dataset.cnv
        unique_entities = X.index.unique()
        y_all = pd.concat(list(dataset[: len(dataset)].values()), axis = 1)

        y = y_all[y_all.drug_name == drug_name]
        y_pred = []

        if y.shape[0] == 0:
            print(f"WARNING: {drug_name} does not exist in dataset")
            return

        for idx, row in y.iterrows():
            y_pred.append(self.main_model(torch.tensor(X.loc[row[dataset.entity_identifier_name]]).view(1, -1)))
        np_out = torch.tensor(y_pred).cpu().detach().numpy()

        y_true = y.copy()
        y_pred = y_true.copy()

        if isinstance(dataset, (AggCategoricalAnnotatedPdxDataset,AggCategoricalAnnotatedTcgaDataset)):
            y_pred["response"] = 1 - np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "response"]
            ].copy()
            measure_column_name = "response"
            if len(np.unique(y_true[measure_column_name])) == 1:
                print(f"Dataset has only 1 class label for {drug_name}")
                return

            drug_roc_auc = roc_auc_score(y_true[measure_column_name], y_pred[measure_column_name])
            aupr = average_precision_score(y_true[measure_column_name], y_pred[measure_column_name])
            fpr, tpr, thresholds = roc_curve(y_true[measure_column_name], y_pred[measure_column_name])
            # generate no skill predictions (majority class)
            c = Counter(y_true[measure_column_name])
            ns_probs = [c.most_common(1)[0][0] for _ in range(len(y_true))]
            ns_auc = roc_auc_score(y_true[measure_column_name], ns_probs)
            ns_fpr, ns_tpr, _ = roc_curve(y_true[measure_column_name], ns_probs)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].plot(fpr, tpr, color="navy", lw= 2, label="Model (area = %0.4f)" % drug_roc_auc)
            axs[0].plot(ns_fpr, ns_tpr, color="grey", linestyle='--', label='No Skill')
            axs[0].set_xlim([0.0, 1.0])
            axs[0].set_ylim([0.0, 1.05])
            axs[0].set_xlabel("False Positive Rate")
            axs[0].set_ylabel("True Positive Rate")
            axs[0].set_title("Receiver Operating Characteristic (ROC) curve")
            axs[0].legend(loc="lower right")

            precision, recall, thresholds = precision_recall_curve(y_true[measure_column_name], y_pred[measure_column_name])
    #         ns_precision, ns_recall, _ = precision_recall_curve(y_true[measure_column_name], ns_probs)
            axs[1].plot(recall, precision, color='purple', lw= 2, label="Model (area = %0.4f)" % aupr)
            axs[1].set_title('Precision-Recall Curve')
            axs[1].set_xlim([0.0, 1.0])
            axs[1].set_ylim([0.0, 1.05])
            axs[1].set_ylabel('Precision')
            axs[1].set_xlabel('Recall')
            no_skill = len(y_true[y_true[measure_column_name]==1]) / len(y_true)
            axs[1].plot([0, 1], [no_skill, no_skill], color="grey", linestyle='--', label='No Skill')
    #         axs[1].plot(ns_recall, ns_precision, color="grey", linestyle='--', label='No Skill')
            axs[1].legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(
                f"Unsupported dataset type - {type(dataset)} - accepted types are [AggCategoricalAnnotatedPdxDataset, AggCategoricalAnnotatedTcgaDataset]"
            )        
        
# instantiate 1 object for each drug 
class VelodromeMainMutCNV(nn.Module):
    def __init__(self, drug_name):
        super(VelodromeMainMutCNV, self).__init__()
        self.drug = drug_name
        self.save_models = f"./velodrome_models/replicate_mut_cnv/tuned/{drug_name}"
        if not os.path.isdir(self.save_models):
            os.makedirs(self.save_models)
        self.save_results = f"./velodrome_results/replicate_mut_cnv/tuned/{drug_name}"
        if not os.path.isdir(self.save_results):
            os.makedirs(self.save_results)
        self.prediction_path = f'./velodrome_predictions/replicate_mut_cnv/tuned/{drug_name}'
        if not os.path.isdir(self.prediction_path):
            os.makedirs(self.prediction_path)
    
    def setup(self, args):
        args['lam2'] = 1 - args['lam1']
        self.args = args
        torch.manual_seed(args['seed'])
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        
        # load cell line train data
        cl_dataset_train = AggCategoricalAnnotatedCellLineDataset(
            is_train=True,
            only_cat_one_drugs=True,
            scale_y=False,
            use_k_best_worst=None,
        )

        # unlabelled TCGA dataset
        tcga_train = AggCategoricalAnnotatedTcgaDataset(
            is_train=True,
            only_cat_one_drugs=True,
        )
        # filter out for the specific drug (train)
        train_y_drug_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == self.drug] # has depmap_id, drug_name, auc 
        train_mut_df = cl_dataset_train.raw_mutations[cl_dataset_train.raw_mutations.index.isin(train_y_drug_df.depmap_id)]

        train_cnv_df = cl_dataset_train.cnv[cl_dataset_train.cnv.index.isin(train_y_drug_df.depmap_id)]
        train_merged_mut_cnv_df = pd.merge(train_mut_df.reset_index(), train_cnv_df.reset_index(), on = "depmap_id").set_index("depmap_id")
        train_merged_df = pd.merge(train_merged_mut_cnv_df.reset_index(), train_y_drug_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")

        df_train, df_val = train_test_split(train_merged_df, test_size=0.2, random_state=42)

        # divide into 2 cell line datasets to be passed into the 2 predictors
        X1_train = df_train.iloc[0: len(df_train)//2].drop("auc", axis = 1).values
        y1_train = df_train.iloc[0: len(df_train)//2]["auc"].values
        X2_train = df_train.iloc[len(df_train)//2:].drop("auc", axis = 1).values
        y2_train = df_train.iloc[len(df_train)//2:]["auc"].values
        self.X1_train = X1_train 

        train1Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X1_train), torch.FloatTensor(y1_train))
        self.trainLoader_1 = torch.utils.data.DataLoader(dataset = train1Dataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        train2Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X2_train), torch.FloatTensor(y2_train))
        self.trainLoader_2 = torch.utils.data.DataLoader(dataset = train2Dataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        X_val = df_val.drop("auc", axis = 1).values
        y_val = df_val.iloc[:]["auc"].values
        self.TX_val = torch.FloatTensor(X_val)
        self.Ty_val = torch.FloatTensor(y_val)
        
#       tcga_train_response_drug_df = tcga_train.tcga_response[tcga_train.tcga_response["drug_name"] == drug_name] # has depmap_id, drug_name, auc 
        tcga_train_response_drug_df = tcga_train.tcga_response # considering all train TCGA entities (if filter for specific drug (docetaxel) get empty tensor for loss1)
        tcga_train_mut_df = tcga_train.raw_mutations[tcga_train.raw_mutations.index.isin(tcga_train_response_drug_df.submitter_id)]

        tcga_train_cnv_df = tcga_train.cnv[tcga_train.cnv.index.isin(tcga_train_response_drug_df.submitter_id)]
        tcga_train_merged_mut_cnv_df = pd.merge(tcga_train_mut_df.reset_index(), tcga_train_cnv_df.reset_index(), on = "submitter_id").set_index("submitter_id")
#       tcga_train_merged_df = pd.merge(tcga_train_merged_mut_cnv_df.reset_index(), tcga_train_response_drug_df.drop("drug_name", axis = 1), on = "submitter_id").set_index("submitter_id")
        X_U = tcga_train_merged_mut_cnv_df.values

        trainUDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_U))
        self.trainULoader = torch.utils.data.DataLoader(dataset = trainUDataset, batch_size=args['bs'], shuffle=True, num_workers=1, drop_last=True)

        print('\n--------------------', self.drug, '--------------------')
        print(f'Cell-line dataset for training Predictor 1: {X1_train.shape}')
        print(f'Cell-line dataset for training Predictor 2: {X2_train.shape}')
        print(f'Total no. of cell-lines for training: {df_train.shape[0]}')
        print(f'TCGA dataset for training both Predictors: {X_U.shape}')
        print(f'Total no. of cell-lines for evaluation: {df_val.shape[0]}')
            
        # instantiate network
        self.model, self.pred1, self.pred2 = Network(args=self.args, X=self.X1_train)
        self.w1 = [torch.tensor(0.5)] * args['epoch']
        self.w2 = [torch.tensor(0.5)] * args['epoch']
        
    def train(self, args): # follows Velodrome.py 
#         self.model, self.pred1, self.pred2 = Network(args=self.args, X=self.X1_train)
        self.setup(args)
        # optimisers
        opt = torch.optim.Adagrad(self.model.parameters(), lr=self.args['lr'], weight_decay = self.args['wd'])
        opt1 = torch.optim.Adagrad(self.pred1.parameters(), lr=self.args['lr1'], weight_decay = self.args['wd1'])
        opt2 = torch.optim.Adagrad(self.pred2.parameters(), lr=self.args['lr2'], weight_decay = self.args['wd2'])
        
        best_pr = -np.inf # allow for negative pearson correlation coefficient
        
        loss_fun = torch.nn.MSELoss()
        total_val = []
        total_aac = []

        train_loss = []
        consistency_loss = []
        covariance_loss = []
        train_pr1 = []
        train_pr2 = []
        val_loss = []
        val_pr = []
    
        train_pred = []
        w1 = []
        w2 = []
        
        for ite in range(self.args['epoch']):
            torch.autograd.set_detect_anomaly(True)
            pred_loss, coral_loss, con_loss, epoch_pr1, epoch_pr2, loss1, loss2 = train(self.args, self.model, self.pred1, self.pred2, loss_fun, opt, opt1, opt2, self.trainLoader_1, self.trainLoader_2, self.trainULoader)

            train_loss.append(pred_loss + coral_loss + con_loss)      
            train_loss.append(pred_loss + con_loss)      
            consistency_loss.append(con_loss)
            covariance_loss.append(coral_loss)
            train_pr1.append(epoch_pr1)
            train_pr2.append(epoch_pr2)

            w1.append(loss1)
            w2.append(loss2)

            epoch_val_loss, epoch_Val_pr,_ = validate_workflow(self.args, self.model, self.pred1, self.pred2, [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))], loss_fun, self.TX_val, self.Ty_val)
            val_loss.append(epoch_val_loss)
            val_pr.append(epoch_Val_pr)                      
#             print(epoch_Val_pr)
            if epoch_Val_pr > best_pr: 
                best_pr = epoch_Val_pr
                torch.save(self.model.state_dict(), os.path.join(self.save_models, 'Best_Model.pt'))
                torch.save(self.pred1.state_dict(), os.path.join(self.save_models, 'Best_Pred1.pt'))
                torch.save(self.pred2.state_dict(), os.path.join(self.save_models, 'Best_Pred2.pt'))

#         plots(self.args, train_loss, consistency_loss, covariance_loss, train_pr1, train_pr2, val_loss, val_pr)
        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))

        self.model.eval()
        self.pred1.eval()
        self.pred2.eval()

        _,_, preds= validate_workflow(self.args, self.model, self.pred1, self.pred2, [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))], loss_fun, self.TX_val, self.Ty_val)
        total_val.append(preds.detach().numpy().flatten())
        total_aac.append(self.Ty_val.detach().numpy())
        self.w1 = w1
        self.w2 = w2

        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))


        final_pred = list(itertools.chain.from_iterable(total_val))
        final_labels = list(itertools.chain.from_iterable(total_aac))  
        dct = {'pr': best_pr}
        print(f'Highest Pearson r achieved during training: {best_pr}')
        return dct
                                   
    def forward(self, X): # follows FuncVelov3's validate_workflow
        self.model.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(self.save_models, 'Best_Pred2.pt')))
        
        self.model.eval()
        self.pred1.eval()
        self.pred2.eval()
        
        ws = [torch.mean(torch.stack(self.w1)), torch.mean(torch.stack(self.w2))]
        w_n = torch.nn.functional.softmax(torch.stack(ws), dim=None)
        w1 = w_n[0]
        w2 = w_n[1]
        
        TX_val = torch.tensor(
            X,
            dtype=torch.float,
        )
        fx_val = self.model(TX_val)
        pred_1 = self.pred1(fx_val)
        pred_2 = self.pred2(fx_val)
        pred_val = w1*pred_1+w2*pred_2 # weighted average
        return pred_val
    
    
# instantiate for running test bed evaluation
class VelodromeTestBedMutCNV():
    def __init__(self, drug_name):
        super(VelodromeTestBedMutCNV, self).__init__()
        self.main_model = VelodromeMainMutCNV(drug_name)
        self.drug_name = drug_name
        
    def train_model(self, best_parameters):
        self.main_model.train(best_parameters)
            
    def tune_model(self, parameters, total_trials):
        best_parameters, values, experiment, model = optimize(
            experiment_name="velodrome_replicate",
            parameters=parameters,
            evaluation_function=self.main_model.train,
            objective_name='pr',
            minimize=False,
            total_trials=total_trials
        )
        return best_parameters, values, experiment, model
                                   
    def get_velodrome_results_df(self, dataset, dataset_name='', save_predictions=False):
        X_mut = dataset.raw_mutations
        X_cnv = dataset.cnv
        X = pd.merge(X_mut.reset_index(), X_cnv.reset_index(), on=dataset.entity_identifier_name).set_index(dataset.entity_identifier_name)
        print(f'X: {X.shape}')
        unique_entities = X.index.unique()
        y_all = pd.concat(list(dataset[: len(dataset)].values()), axis = 1)
        drug_name = self.drug_name
        
        y = y_all[y_all.drug_name == drug_name]
        y_pred = []

        if y.shape[0] == 0:
            print(f"WARNING: {drug_name} does not exist in dataset")
            return pd.DataFrame()

        for idx, row in y.iterrows():
            y_pred.append(self.main_model(torch.tensor(X.loc[row[dataset.entity_identifier_name]]).view(1, -1)))
        np_out = torch.tensor(y_pred).cpu().detach().numpy()

        y_true = y.copy()
        y_pred = y_true.copy()

        if isinstance(dataset, (AggCategoricalAnnotatedPdxDataset, AggCategoricalAnnotatedTcgaDataset)):
            y_pred["response"] = 1 - np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "response"]
            ].copy()
            measure_column_name = "response"
            if len(np.unique(y_true[measure_column_name])) == 1:
                print(f"Dataset has only 1 class label for {drug_name}")
                drug_roc_auc = np.nan
            else:
                drug_roc_auc = roc_auc_score(y_true[measure_column_name], y_pred[measure_column_name])
            drug_spearmanr = spearmanr(y_true[measure_column_name], y_pred[measure_column_name])
            if len(y_true) < 2:
                drug_pearsonr = (np.nan, np.nan)
            else:
                drug_pearsonr = pearsonr(y_true[measure_column_name], y_pred[measure_column_name])

            drug_results_df = pd.DataFrame(data={'drug_name': [drug_name],
                                                 'length_dataset': [len(y_true)],
                                                'roc_auc': [drug_roc_auc], 
                                                'aupr': [average_precision_score(y_true[measure_column_name], y_pred[measure_column_name])], 
                                                'spearmanr_correlation': [drug_spearmanr[0]], 
                                                'spearmanr_pvalue': [drug_spearmanr[1]], 
                                                'pearsonr_correlation': [drug_pearsonr[0]],
                                                'pearsonr_pvalue': [drug_pearsonr[1]]})

        elif isinstance(dataset, AggCategoricalAnnotatedCellLineDataset):
            y_pred["auc"] = np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "auc"]
            ].copy()
            measure_column_name = "auc"
            drug_spearmanr = spearmanr(y_true[measure_column_name], y_pred[measure_column_name])
            drug_pearsonr = pearsonr(y_true[measure_column_name], y_pred[measure_column_name])
            drug_results_df = pd.DataFrame(data={'drug_name': [drug_name],
                                                 'length_dataset': [len(y_true)],
                                                'spearmanr_correlation': [drug_spearmanr[0]], 
                                                'spearmanr_pvalue': [drug_spearmanr[1]], 
                                                'pearsonr_correlation': [drug_pearsonr[0]],
                                                'pearsonr_pvalue': [drug_pearsonr[1]]})
        else:
            raise ValueError(
                f"Unsupported dataset type - {type(dataset)} - accepted types are [AggCategoricalAnnotatedCellLineDataset, AggCategoricalAnnotatedPdxDataset, AggCategoricalAnnotatedTcgaDataset]"
            )    
            
        if save_predictions:
            np.savetxt(os.path.join(self.main_model.prediction_path, f'{dataset_name}_testset_groundtruth.csv'), y_true[measure_column_name])
            np.savetxt(os.path.join(self.main_model.prediction_path, f'{dataset_name}_testset_predictions.csv'), y_pred[measure_column_name])

        return drug_results_df
    
    def get_roc_pr_curves(self, drug_name, dataset):
        X_mut = dataset.raw_mutations
        X_cnv = dataset.cnv
        X = pd.merge(X_mut.reset_index(), X_cnv.reset_index(), on=dataset.entity_identifier_name).set_index(dataset.entity_identifier_name)
        print(f'X: {X.shape}')
        unique_entities = X.index.unique()
        y_all = pd.concat(list(dataset[: len(dataset)].values()), axis = 1)

        y = y_all[y_all.drug_name == drug_name]
        y_pred = []

        if y.shape[0] == 0:
            print(f"WARNING: {drug_name} does not exist in dataset")
            return

        for idx, row in y.iterrows():
            y_pred.append(self.main_model(torch.tensor(X.loc[row[dataset.entity_identifier_name]]).view(1, -1)))
        np_out = torch.tensor(y_pred).cpu().detach().numpy()

        y_true = y.copy()
        y_pred = y_true.copy()

        if isinstance(dataset, (AggCategoricalAnnotatedPdxDataset,AggCategoricalAnnotatedTcgaDataset)):
            y_pred["response"] = 1 - np_out.squeeze()
            y_pred = y_pred[
                [dataset.entity_identifier_name, "drug_name", "response"]
            ].copy()
            measure_column_name = "response"
            if len(np.unique(y_true[measure_column_name])) == 1:
                print(f"Dataset has only 1 class label for {drug_name}")
                return

            drug_roc_auc = roc_auc_score(y_true[measure_column_name], y_pred[measure_column_name])
            aupr = average_precision_score(y_true[measure_column_name], y_pred[measure_column_name])
            fpr, tpr, thresholds = roc_curve(y_true[measure_column_name], y_pred[measure_column_name])
            # generate no skill predictions (majority class)
            c = Counter(y_true[measure_column_name])
            ns_probs = [c.most_common(1)[0][0] for _ in range(len(y_true))]
            ns_auc = roc_auc_score(y_true[measure_column_name], ns_probs)
            ns_fpr, ns_tpr, _ = roc_curve(y_true[measure_column_name], ns_probs)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].plot(fpr, tpr, color="navy", lw= 2, label="Model (area = %0.4f)" % drug_roc_auc)
            axs[0].plot(ns_fpr, ns_tpr, color="grey", linestyle='--', label='No Skill')
            axs[0].set_xlim([0.0, 1.0])
            axs[0].set_ylim([0.0, 1.05])
            axs[0].set_xlabel("False Positive Rate")
            axs[0].set_ylabel("True Positive Rate")
            axs[0].set_title("Receiver Operating Characteristic (ROC) curve")
            axs[0].legend(loc="lower right")

            precision, recall, thresholds = precision_recall_curve(y_true[measure_column_name], y_pred[measure_column_name])
    #         ns_precision, ns_recall, _ = precision_recall_curve(y_true[measure_column_name], ns_probs)
            axs[1].plot(recall, precision, color='purple', lw= 2, label="Model (area = %0.4f)" % aupr)
            axs[1].set_title('Precision-Recall Curve')
            axs[1].set_xlim([0.0, 1.0])
            axs[1].set_ylim([0.0, 1.05])
            axs[1].set_ylabel('Precision')
            axs[1].set_xlabel('Recall')
            no_skill = len(y_true[y_true[measure_column_name]==1]) / len(y_true)
            axs[1].plot([0, 1], [no_skill, no_skill], color="grey", linestyle='--', label='No Skill')
    #         axs[1].plot(ns_recall, ns_precision, color="grey", linestyle='--', label='No Skill')
            axs[1].legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(
                f"Unsupported dataset type - {type(dataset)} - accepted types are [AggCategoricalAnnotatedPdxDataset, AggCategoricalAnnotatedTcgaDataset]] "
            )    
        