import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch
import random

from torch import nn
from torch.nn import functional as F
from functools import cached_property
import pandas as pd
import numpy as np
from torch.nn import Linear, ReLU, Sequential
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split

from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

import ax
from ax import RangeParameter, ChoiceParameter, FixedParameter
from ax import ParameterType, SearchSpace
from ax.service.managed_loop import optimize

from DecotaDataHelpers import (
    get_dataset_decota
)

class FeatModule(nn.Module): 
    def __init__(self, flags, X): # X as in Network(args, X)
        IE_dim = X
        
        super(FeatModule, self).__init__()
        if flags['hd'] == 1:
            self.features = torch.nn.Sequential(nn.Linear(IE_dim, 512),
                                                nn.BatchNorm1d(512),
                                                nn.ReLU(),
                                                nn.Dropout(flags['ldr']),
                                                nn.Linear(512, 128),
                                                nn.Sigmoid())
        elif flags['hd'] == 2:
            self.features = torch.nn.Sequential(nn.Linear(IE_dim, 256),
                                                nn.BatchNorm1d(256),
                                                nn.ReLU(),
                                                nn.Dropout(flags['ldr']),
                                                nn.Linear(256, 256),
                                                nn.Sigmoid()
                                                ) 
            
        elif flags['hd'] == 3:
             self.features = torch.nn.Sequential(nn.Linear(IE_dim, 128),
                                                nn.BatchNorm1d(128),
                                                nn.ReLU(),
                                                nn.Dropout(flags['ldr']),
                                                nn.Linear(128, 128),
                                                nn.BatchNorm1d(128),
                                                nn.ReLU(),
                                                nn.Dropout(flags['ldr']),
                                                nn.Linear(128, 128),
                                                nn.Sigmoid()
                                                )      
            
        elif flags['hd'] == 4:
            self.features = torch.nn.Sequential(nn.Linear(IE_dim, 64),
                                                nn.BatchNorm1d(64),
                                                nn.ReLU(),
                                                nn.Dropout(flags['ldr']),
                                                nn.Linear(64, 64),
                                                nn.BatchNorm1d(64),
                                                nn.ReLU(),
                                                nn.Dropout(flags['ldr']),
                                                nn.Linear(64, 64),
                                                nn.Sigmoid()
                                                )   

    def forward(self, x):
        out = self.features(x)
        return out    


class ClassifierModule(nn.Module):
    def __init__(self, flags):
        super(ClassifierModule, self).__init__()        
        if flags['hd'] == 1: 
                dim = 128            
        if flags['hd'] == 2: 
            dim = 256
        if flags['hd'] == 3:
            dim = 128
        if flags['hd'] == 4:
            dim = 64               
        self.pred = torch.nn.Sequential(
            nn.Linear(dim, 1)) 

    def forward(self, x):
        out = self.pred(x) 
#         end_points = {'Predictions': F.softmax(input=x, dim=-1)}
        end_points = {'Predictions': torch.sigmoid(input=out)}
        return out, end_points
    
class DomainSpecificNN(nn.Module):
    def __init__(self, flags, X):
        super(DomainSpecificNN, self).__init__()

        self.feature = FeatModule(flags, X)
        self.classifier = ClassifierModule(flags)

    def forward(self, x): # domain is indexed 
        net = self.feature(x)
        net, end_points = self.classifier(net)
        return net, end_points
    
def shuffle_data(samples, labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_samples, shuffled_labels

def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()
    
class BatchGenerator:
    def __init__(self, flags, stage, file_path):
        # file_path is Pandas dataframe containing specific domain's data for a specific drug

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(flags, stage, file_path)
        self.load_data()

    def configuration(self, flags, stage, file_path):
        self.batch_size = flags['batch_size']
        self.current_index = -1
        self.file_path = file_path
        self.stage = stage

    def load_data(self):
        file_path = self.file_path
        self.xdata = file_path.drop("response", axis = 1).values
        self.labels = file_path.iloc[:]["response"].values
        
        self.file_num_train = len(self.labels)
        
        if self.stage == 'train':
            self.xdata, self.labels = shuffle_data(samples=self.xdata, labels=self.labels)

    def get_xdata_labels_batch(self):

        xdata = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.xdata, self.labels = shuffle_data(samples=self.xdata, labels=self.labels)

            xdata.append(self.xdata[self.current_index])
            labels.append(self.labels[self.current_index])

        xdata = np.stack(xdata)
        labels = np.stack(labels)

        return xdata, labels
    
class ModelDECOTAwoMixup():
    def __init__(self, drug_name, input_name):
        self.drug_name = drug_name
        self.input_name = input_name
        self.model_path = f'./decota_models/wo_mixup/{input_name}/{drug_name}/tuned/'
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.logs = f'./decota_logs/wo_mixup/{input_name}/{drug_name}/tuned/'
        if not os.path.isdir(self.logs):
            os.makedirs(self.logs)
        self.prediction_path = f'./decota_predictions/wo_mixup/{input_name}/{drug_name}/tuned/'
        if not os.path.isdir(self.prediction_path):
            os.makedirs(self.prediction_path)
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    def setup(self, flags):    
        # fix all seeds
        random.seed(flags['seed'])
        np.random.seed(flags['seed'])
        torch.manual_seed(flags['seed'])
        torch.cuda.manual_seed_all(flags['seed'])
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        if self.input_name == 'mut_cnv':
            self.source_nn = DomainSpecificNN(flags, 648).to(self.device)
            self.target_nn = DomainSpecificNN(flags, 648).to(self.device)
        else:
            self.source_nn = DomainSpecificNN(flags, 324).to(self.device)
            self.target_nn = DomainSpecificNN(flags, 324).to(self.device)

        print(self.source_nn)
        print(self.target_nn)
        print('flags:', flags)

        flags_log = os.path.join(self.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.configure(flags)

        self.best_auroc_val = float('-inf')

    def setup_path(self, flags):        
        drug_name = self.drug_name
        
        self.train_paths = ['cl',
                          'tcga']
        
        self.val_paths = [
#                         'cl',
                          'tcga']
        
        # get test data for particular drug
        self.test_paths = [
#                         'cl',
                          'pdx',
                          'tcga']
        
        temp_paths = self.train_paths.copy()
        self.train_paths = []
        print(f'-------{drug_name}-------')
        for t in temp_paths:
            train_df = get_dataset_decota(t, drug_name, self.input_name, 'train')
            self.train_paths.append(train_df)
            print(f'{t} train set size: {len(train_df)}')
            
        temp_paths = self.val_paths.copy()
        self.val_paths = []
        for t in temp_paths:
            val_df = get_dataset_decota(t, drug_name, self.input_name, 'val')
            self.val_paths.append(val_df)
            print(f'{t} validation set size: {len(val_df)}')
            
        temp_paths = self.test_paths.copy()
        self.test_paths = []
        for t in temp_paths:
            test_df = get_dataset_decota(t, drug_name, self.input_name, 'test')
            self.test_paths.append(test_df)
            print(f'{t} test set size: {len(test_df)}')
        
        # for training domain specific 
        self.batGenTrains = []
        for train_path in self.train_paths: 
            # each domain dataset has a BatchImageGenerator object, call get_images_labels_batch to get batch
            batGenTrain = BatchGenerator(flags=flags, file_path=train_path, stage='train')
            self.batGenTrains.append(batGenTrain)

        # for test_workflow
        self.batGenVals = []
        for val_path in self.val_paths:
            batGenVal = BatchGenerator(flags=flags, file_path=val_path, stage='val')
            self.batGenVals.append(batGenVal)
        
        # for test_workflow
        self.batGenTests = []
        for test_path in self.test_paths:
            batGenTest = BatchGenerator(flags=flags, file_path=test_path, stage='test')
            self.batGenTests.append(batGenTest)
            
        # unlabelled TCGA data
        unlabelled_df = get_dataset_decota('unlabelled_tcga', drug_name, self.input_name)
        self.batGenUnlabelled = BatchGenerator(flags=flags, file_path=unlabelled_df, stage='train')
        print(f'unlabelled TCGA set: {len(unlabelled_df)}')
        
    def configure(self, flags):
        self.optimizer_source = optim.SGD(params=[{'params': self.source_nn.feature.parameters()},
                                             {'params': self.source_nn.classifier.parameters()}],
                                 lr=flags['lr'],
                                 weight_decay=flags['weight_decay'],
                                 momentum=flags['momentum'])

        self.scheduler_source = lr_scheduler.StepLR(optimizer=self.optimizer_source,
                                                 step_size=flags['step_size'], gamma=0.1)

        self.optimizer_target = optim.SGD(params=[{'params': self.target_nn.feature.parameters()},
                                             {'params': self.target_nn.classifier.parameters()}],
                                 lr=flags['lr'],
                                 weight_decay=flags['weight_decay'],
                                 momentum=flags['momentum'])

        self.scheduler_target = lr_scheduler.StepLR(optimizer=self.optimizer_target,
                                                 step_size=flags['step_size'], gamma=0.1)

#         self.loss_fn = crossentropyloss().cuda()
        self.classification_loss_fn = nn.BCEWithLogitsLoss().to(self.device) # inputs do not need to go through sigmoid prior to input, like how crossentropyloss does not need to be softmax activated
        self.regression_loss_fn = nn.MSELoss().to(self.device)
            
    def warm_target_nn(self, flags):
        # train on source domain data
        self.target_nn.train()
        index = 1
        
        # get the inputs and labels from the data reader
        ent_loss = 0.0
        
        xdata_train, labels_train = self.batGenTrains[index].get_xdata_labels_batch()

        inputs, labels = torch.from_numpy(
            np.array(xdata_train, dtype=np.float32)), torch.from_numpy(
            np.array(labels_train, dtype=np.float32))

        # wrap the inputs and labels in Variable
        inputs, labels = Variable(inputs, requires_grad=False).to(self.device), \
                         Variable(labels, requires_grad=False).long().to(self.device)

        # forward
        outputs, _ = self.target_nn(x=inputs) 

        # loss
        loss = self.classification_loss_fn(outputs.squeeze(), labels.float())

        ent_loss += loss

        self.optimizer_target.zero_grad()

        # backward your network
        ent_loss.backward()

        # optimize the parameters
        self.optimizer_target.step()
        
        # change order due to pytorch ver
        self.scheduler_target.step()

        flags_log = os.path.join(self.logs, 'target_nn_loss_log.txt')
        write_log(str(ent_loss.item()), flags_log)

    def warm_source_nn(self, flags):
        # train on source domain data
        self.source_nn.train()
        index = 0
        
        # get the inputs and labels from the data reader
        ent_loss = 0.0
        
        xdata_train, labels_train = self.batGenTrains[index].get_xdata_labels_batch()

        inputs, labels = torch.from_numpy(
            np.array(xdata_train, dtype=np.float32)), torch.from_numpy(
            np.array(labels_train, dtype=np.float32))

        # wrap the inputs and labels in Variable
        inputs, labels = Variable(inputs, requires_grad=False).to(self.device), \
                         Variable(labels, requires_grad=False).long().to(self.device)

        # forward
        outputs, _ = self.source_nn(x=inputs) 

        # loss
        loss = self.regression_loss_fn(outputs.squeeze(), labels.float())

        ent_loss += loss

        self.optimizer_source.zero_grad()

        # backward your network
        ent_loss.backward()

        # optimize the parameters
        self.optimizer_source.step()
        
        # change order due to pytorch ver
        self.scheduler_source.step()

        flags_log = os.path.join(self.logs, 'source_nn_loss_log.txt')
        write_log(str(ent_loss.item()), flags_log)
        
    def zero_grad_all(self):
        self.optimizer_source.zero_grad()
        self.optimizer_target.zero_grad()
        
    def cotraining_nn(self, ite, flags): # ite = num_iter in training loop
        self.source_nn.train()
        self.target_nn.train()
        
        target_nn_loss = 0.0
        # calculate loss of passing train target data to target_model
        self.zero_grad_all()
        index = 1
        xdata_train, labels_train = self.batGenTrains[index].get_xdata_labels_batch()
        inputs, labels = torch.from_numpy(
            np.array(xdata_train, dtype=np.float32)), torch.from_numpy(
            np.array(labels_train, dtype=np.float32))
        inputs, labels = Variable(inputs, requires_grad=False).to(self.device), \
                         Variable(labels, requires_grad=False).long().to(self.device)
        outputs, _ = self.target_nn(x=inputs) 
        loss = self.classification_loss_fn(outputs.squeeze(), labels.float())
        target_nn_loss += loss
        self.zero_grad_all()
        
#         print("---- pass train target data to target_nn ----")
#         print("inputs", inputs.size())
#         print("outputs", outputs.size())
#         print("labels", labels.size())
        
        source_nn_loss = 0.0
        # calculate loss of passing train source data to source_model
        self.zero_grad_all()
        index = 0
        xdata_train, labels_train = self.batGenTrains[index].get_xdata_labels_batch()
        inputs, labels = torch.from_numpy(
            np.array(xdata_train, dtype=np.float32)), torch.from_numpy(
            np.array(labels_train, dtype=np.float32))
        inputs, labels = Variable(inputs, requires_grad=False).to(self.device), \
                         Variable(labels, requires_grad=False).long().to(self.device)
        outputs, _ = self.source_nn(x=inputs) 
        loss = self.regression_loss_fn(outputs.squeeze(), labels.float())
        source_nn_loss += loss
        self.zero_grad_all()
        
#         print("---- pass train source data to source_nn ----")
#         print("inputs", inputs.size())
#         print("outputs", outputs.size())
#         print("labels", labels.size())
        
        # pseudo-label
        xdata_train, _ = self.batGenUnlabelled.get_xdata_labels_batch()
        inputs = torch.from_numpy(np.array(xdata_train, dtype=np.float32))
        inputs = Variable(inputs, requires_grad=False).to(self.device)
        
        outputs, predictions = self.target_nn(x=inputs) 
        class_predictions = predictions['Predictions'].max(1)
        mask = class_predictions[0] >= flags['predicted_probability_conf']
        pseudo_samples_inputs_by_tnn = inputs[mask]
        pseudo_predicted_probs_by_tnn = class_predictions[0][mask]
        
        outputs, _ = self.source_nn(x=inputs) 
        predicted_aadrc = outputs.max(1)
        mask = predicted_aadrc[0] >= flags['predicted_aadrc_conf']
        pseudo_samples_inputs_by_snn = inputs[mask]
        pseudo_predicted_aadrc_by_snn = predicted_aadrc[0][mask]
        
        # train target nn by regressing on predicted AADRC from source nn
        # more than 1 sample needed bc of batch norm
        if pseudo_samples_inputs_by_snn.size(0) > 1:
            outputs, predictions = self.target_nn(x=pseudo_samples_inputs_by_snn) 
            loss = self.regression_loss_fn(predictions['Predictions'].squeeze(), pseudo_predicted_aadrc_by_snn.squeeze())
            target_nn_loss += loss
            self.zero_grad_all()
            target_nn_loss.backward(retain_graph=True)
            
#             print("---- pass pseudo labelled data by SNN ----")
#             print("inputs", pseudo_samples_inputs_by_snn.size())
#             print("outputs", outputs.size())
#             print("labels", pseudo_predicted_aadrc_by_snn.size())
            
        self.zero_grad_all()
        # train source nn by regressing on predicted probabilities from target nn
        if pseudo_samples_inputs_by_tnn.size(0) > 1:
            outputs, _ = self.source_nn(x=pseudo_samples_inputs_by_tnn) 
            loss = self.regression_loss_fn(outputs.squeeze(), pseudo_predicted_probs_by_tnn.squeeze())
            source_nn_loss += loss
            self.zero_grad_all()
            source_nn_loss.backward()
            
#             print("---- pass pseudo labelled data by TNN ----")
#             print("inputs", pseudo_samples_inputs_by_tnn.size())
#             print("outputs", outputs.size())
#             print("labels", pseudo_predicted_probs_by_tnn.size())
        
        # to avoid 'RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation'
        if pseudo_samples_inputs_by_snn.size(0) > 1:
            self.optimizer_target.step()
            self.scheduler_target.step()
        if pseudo_samples_inputs_by_tnn.size(0) > 1:
            self.optimizer_source.step()
            self.scheduler_source.step()
        self.zero_grad_all()
        
#         flags_log = os.path.join(self.logs, 'agg_ent_loss.txt')
#         write_log(str(agg_ent_loss.item()), flags_log)
#         flags_log = os.path.join(self.logs, 'epir_loss.txt')
#         write_log(str(epir_loss.item()), flags_log)
#         if ite >= flags['ite_train_epi_c']:
#             flags_log = os.path.join(self.logs, 'epic_loss.txt')
#             write_log(str(epic_loss.item()), flags_log)
#         if ite >= flags['ite_train_epi_f']:
#             flags_log = os.path.join(self.logs, 'epif_loss.txt')
#             write_log(str(epif_loss.item()), flags_log)

    def test(self, flags, ite, log_prefix='', log_dir='epi_logs/', batGenTest=None):
        # switch on the network test mode
        self.source_nn.eval()
        self.target_nn.eval()
        
        xdata_test = batGenTest.xdata
        labels_test = batGenTest.labels
        xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
        
        if flags['use_source'] == 1:
            # ensemble: take average of predicted AADRC and predicted probability
            outputs, predictions = self.target_nn(x=xdata_test) 
            predicted_probabilities = predictions['Predictions']
            predicted_aadrc, _ = self.source_nn(x=xdata_test) 
            result_predictions = torch.stack([predicted_aadrc, predicted_probabilities]).mean(dim=0)
            
        else:
            outputs, predictions = self.target_nn(x=xdata_test) 
            result_predictions = predictions['Predictions']
            
        if len(np.unique(labels_test)) == 1:
            roc_auc = np.nan
        else:
            roc_auc = roc_auc_score(labels_test, result_predictions.cpu().data.detach().numpy())
#         rounded_predictions = torch.round(predictions).cpu().data.detach().numpy()
#         acc = accuracy_score(y_true=labels_test, y_pred=rounded_predictions)
#         print('----------AUROC test----------:', roc_auc)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, AUROC:{}\n'.format(ite, roc_auc))
        f.close()

        # switch on the network train mode
        self.source_nn.train()
        self.target_nn.train()
#         self.bn_process(flags)

        return roc_auc

    def test_workflow(self, batGenVals, flags, ite, prefix=''):

        auroc_lst = []
        for count, batGenVal in enumerate(batGenVals):
            auroc_val = self.test(batGenTest=batGenVal, flags=flags, ite=ite)
#                                      log_dir=flags['logs'], log_prefix='val_index_{}'.format(count))
            if pd.isna(auroc_val):
#                 if len(np.unique(batGenVal.labels)) == 1:
#                     print(f"Unable to calculate AUROC for validation set {count} due to only 1 unique class")
#                 else:
#                     raise ValueError(
#                         f"Unable to calculate AUROC for validation set {count}"
#                     )
                if len(np.unique(batGenVal.labels)) != 1:
                    raise ValueError(
                        f"Unable to calculate AUROC for validation set {count}"
                    ) 
            else:
                auroc_lst.append(auroc_val)
        mean_auroc = np.mean(auroc_lst)

#         if mean_auroc > self.best_auroc_val:
#             self.best_auroc_val = mean_auroc
            
#             for count, batGenTest in enumerate(self.batGenTests):
#                 auroc_test = self.test(batGenTest=batGenTest, flags=flags, ite=ite,
#                                          log_dir=flags['logs'], log_prefix='dg_test_{}'.format(count))

#                 f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
#                 f.write(
#                     'ite:{}, best val AUROC:{}, test_{} AUROC:{}\n'.format(ite, self.best_auroc_val, 
#                                                                            count, auroc_test))
#                 f.close()
                
#             torch.save(self.agg_nn.state_dict(), os.path.join(flags['model_path'], 'Best_Model.pt'))
        return mean_auroc



    def train_evaluate(self, flags, save_model=False):
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)
        
        self.source_nn.train()
        self.target_nn.train()
#         self.bn_process(flags) # set to eval all batch norms
        
        max_auroc = float('-inf')
        for ite in range(flags['loops_train']):
            if ite <= flags['loops_warm']: # warm up using corresponding training data
                self.warm_target_nn(flags)
                self.warm_source_nn(flags)

                if (ite % flags['test_every'] == 0) and (ite != 0):
                    test_auroc = self.test_workflow(self.batGenVals, flags, ite, prefix='')
            else: # co-training
                self.cotraining_nn(ite, flags)

#                 if (ite % flags['test_every'] == 0) and (ite != 0):
                test_auroc = self.test_workflow(self.batGenVals, flags, ite, prefix='')
                if test_auroc > max_auroc:
                    max_auroc = test_auroc
                    if save_model:
                        torch.save(self.source_nn.state_dict(), os.path.join(self.model_path, f'Best_Source_NN.pt'))
                        torch.save(self.target_nn.state_dict(), os.path.join(self.model_path, f'Best_Target_NN.pt'))

        dct = {'auc': max_auroc}
        print(f'Highest AUROC achieved during training: {max_auroc}')
        return dct
    
    def tune_model(self, parameters, total_trials):
        best_parameters, values, experiment, model = optimize(
            experiment_name="decota_without_mixup",
            parameters=parameters,
            evaluation_function=self.train_evaluate,
            objective_name='auc',
            minimize=False,
            total_trials=total_trials
        )
        return best_parameters, values, experiment, model
                    
    def get_results(self, best_parameters):
        # switch on the network test mode
        self.train_evaluate(best_parameters, save_model=True)
        self.source_nn.load_state_dict(torch.load(os.path.join(self.model_path, f'Best_Source_NN.pt')))
        self.target_nn.load_state_dict(torch.load(os.path.join(self.model_path, f'Best_Target_NN.pt')))
        self.source_nn.eval()
        self.target_nn.eval()
        
        domain_names_dct = {
#             0: 'cell-line',
            0: 'PDX',
            1: 'TCGA'
        }
        results_df = pd.DataFrame()
        for count, batGenTest in enumerate(self.batGenTests):
            xdata_test = batGenTest.xdata
            labels_test = batGenTest.labels
            if len(labels_test) < 1:
                print(f'{domain_names_dct[count]} test set not available')
                continue
            response_value_counts = pd.value_counts(labels_test).to_dict()
            xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
            
            if best_parameters['use_source'] == 1:
                # ensemble: take average of predicted AADRC and predicted probability
                outputs, target_nn_predictions = self.target_nn(x=xdata_test) 
                predicted_probabilities = target_nn_predictions['Predictions']
                predicted_aadrc, _ = self.source_nn(x=xdata_test) 
                average_predictions = torch.stack([predicted_aadrc, predicted_probabilities]).mean(dim=0)
                predictions = average_predictions.cpu().data.detach().numpy().flatten()
            else:
                outputs, target_nn_predictions = self.target_nn(x=xdata_test) 
                predicted_probabilities = target_nn_predictions['Predictions']
                predictions = predicted_probabilities.cpu().data.detach().numpy().flatten()
            
            np.savetxt(os.path.join(self.prediction_path, f'{domain_names_dct[count]}_testset_groundtruth.csv'), labels_test)
            np.savetxt(os.path.join(self.prediction_path, f'{domain_names_dct[count]}_testset_predictions.csv'), predictions)
            
            if len(np.unique(labels_test)) == 1:
                print(f"Test set Idx {count} has only response {np.unique(labels_test)}")
                drug_roc_auc = np.nan
            else:
                drug_roc_auc = roc_auc_score(labels_test, predictions)
            print(np.quantile(predictions, [0, 0.10, 0.25,0.5,0.75,1]))

    #         print('After Rounding', predictions)
    #         rounded_predictions = torch.round(predictions).cpu().data.detach().numpy()
    #         accuracy = accuracy_score(y_true=labels_test, y_pred=rounded_predictions)
    #         print('Ouputs', raw_outputs)
    #         print('Labels', labels_test)
    
            drug_df = pd.DataFrame(data={'dataset': [domain_names_dct[count]],
                                         'length_dataset': [labels_test.shape[0]],
                                         'value_counts': [response_value_counts],
                                         'roc_auc': [drug_roc_auc], 
                                         'aupr': [average_precision_score(labels_test, predictions)], })
            results_df = pd.concat([results_df, drug_df])
        # switch on the network train mode
        self.source_nn.train()
        self.target_nn.train()
#         self.bn_process(best_parameters)
        return results_df
    
    def get_train_set_results(self, best_parameters, train_model):
        # switch on the network test mode
        if train_model:
            self.train_evaluate(best_parameters, save_model=True)
        self.source_nn.load_state_dict(torch.load(os.path.join(self.model_path, f'Best_Source_NN.pt')))
        self.target_nn.load_state_dict(torch.load(os.path.join(self.model_path, f'Best_Target_NN.pt')))
        self.source_nn.eval()
        self.target_nn.eval()
        
        # get test data for particular drug
        train_all_paths = ['pdx',
                          'tcga']
        train_all_datasets = []
        for t in train_all_paths:
            train_all_datasets.append(get_dataset_decota(t, self.drug_name, self.input_name, 'train_all'))
        
        domain_names_dct = {
#             0: 'cell-line',
            0: 'PDX',
            1: 'TCGA'
        }
        results_df = pd.DataFrame()
        for count, batGenTest in enumerate(train_all_datasets):
            xdata_test = batGenTest.drop("response", axis = 1).values
            labels_test = batGenTest.iloc[:]["response"].values
            response_value_counts = pd.value_counts(labels_test).to_dict()

            xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
            
            if best_parameters['use_source'] == 1:
                # ensemble: take average of predicted AADRC and predicted probability
                outputs, target_nn_predictions = self.target_nn(x=xdata_test) 
                predicted_probabilities = target_nn_predictions['Predictions']
                predicted_aadrc, _ = self.source_nn(x=xdata_test) 
                average_predictions = torch.stack([predicted_aadrc, predicted_probabilities]).mean(dim=0)
                predictions = average_predictions.cpu().data.detach().numpy().flatten()
            else:
                outputs, target_nn_predictions = self.target_nn(x=xdata_test) 
                predicted_probabilities = target_nn_predictions['Predictions']
                predictions = predicted_probabilities.cpu().data.detach().numpy().flatten()
            
#             print('After Sigmoid', predictions)
            if len(np.unique(labels_test)) <= 1:
                print(f"Test set Idx {count} has only response {np.unique(labels_test)}")
                drug_roc_auc = np.nan
                drug_aupr = np.nan
            else:
                drug_roc_auc = roc_auc_score(labels_test, predictions)
                drug_aupr = average_precision_score(labels_test, predictions)
            
    #         print('After Rounding', predictions)
    #         rounded_predictions = torch.round(predictions).cpu().data.detach().numpy()
    #         accuracy = accuracy_score(y_true=labels_test, y_pred=rounded_predictions)
    #         print('Ouputs', raw_outputs)
    #         print('Labels', labels_test)
    
            drug_df = pd.DataFrame(data={'dataset': [domain_names_dct[count]],
                                         'length_dataset': [labels_test.shape[0]],
                                         'value_counts': [response_value_counts],
                                         'roc_auc': [drug_roc_auc], 
                                         'aupr': [drug_aupr], })
            results_df = pd.concat([results_df, drug_df])
        # switch on the network train mode
        self.source_nn.train()
        self.target_nn.train()
#         self.bn_process(best_parameters)
        return results_df
    
    def get_roc_pr_curves(self, best_parameters, train_model, dataset):
        # switch on the network test mode
        if train_model:
            self.train_evaluate(best_parameters, save_model=True)
        self.source_nn.load_state_dict(torch.load(os.path.join(self.model_path, 'Best_Source_NN.pt')))
        self.target_nn.load_state_dict(torch.load(os.path.join(self.model_path, 'Best_Target_NN.pt')))
        self.source_nn.eval()
        self.target_nn.eval()

        domain_idx_dct = {
#             'cl': 0,
            'pdx': 0,
            'tcga': 1
        }
        batGenTest = self.batGenTests[domain_idx_dct[dataset]]
        xdata_test = batGenTest.xdata
        labels_test = batGenTest.labels

        xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
        
        if best_parameters['use_source'] == 1:
            # ensemble: take average of predicted AADRC and predicted probability
            outputs, target_nn_predictions = self.target_nn(x=xdata_test) 
            predicted_probabilities = target_nn_predictions['Predictions']
            predicted_aadrc, _ = self.source_nn(x=xdata_test) 
            average_predictions = torch.stack([predicted_aadrc, predicted_probabilities]).mean(dim=0)
            predictions = average_predictions.cpu().data.detach().numpy().flatten()
        else:
            outputs, target_nn_predictions = self.target_nn(x=xdata_test) 
            predicted_probabilities = target_nn_predictions['Predictions']
            predictions = predicted_probabilities.cpu().data.detach().numpy().flatten()
            
    #             print('After Sigmoid', predictions)
        if len(np.unique(labels_test)) == 1:
            print(f"Test set Idx {count} has only response {np.unique(labels_test)}")
            return

        drug_roc_auc = roc_auc_score(labels_test, predictions)
        aupr = average_precision_score(labels_test, predictions)
        fpr, tpr, thresholds = roc_curve(labels_test, predictions)

        # generate no skill predictions (majority class)
        c = Counter(labels_test)
        ns_probs = [c.most_common(1)[0][0] for _ in range(len(labels_test))]
        ns_auc = roc_auc_score(labels_test, ns_probs)
        ns_fpr, ns_tpr, _ = roc_curve(labels_test, ns_probs)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(fpr, tpr, color="navy", lw= 2, label="Model (area = %0.4f)" % drug_roc_auc)
        axs[0].plot(ns_fpr, ns_tpr, color="grey", linestyle='--', label='No Skill')
        axs[0].set_xlim([0.0, 1.0])
        axs[0].set_ylim([0.0, 1.05])
        axs[0].set_xlabel("False Positive Rate")
        axs[0].set_ylabel("True Positive Rate")
        axs[0].set_title("Receiver Operating Characteristic (ROC) curve")
        axs[0].legend(loc="lower right")

        precision, recall, thresholds = precision_recall_curve(labels_test, predictions)
    #         ns_precision, ns_recall, _ = precision_recall_curve(y_true[measure_column_name], ns_probs)
        axs[1].plot(recall, precision, color='purple', lw= 2, label="Model (area = %0.4f)" % aupr)
        axs[1].set_title('Precision-Recall Curve')
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].set_ylabel('Precision')
        axs[1].set_xlabel('Recall')
        no_skill = sum(np.where(labels_test == 1, 1, 0)) / len(labels_test)
        axs[1].plot([0, 1], [no_skill, no_skill], color="grey", linestyle='--', label='No Skill')
    #         axs[1].plot(ns_recall, ns_precision, color="grey", linestyle='--', label='No Skill')
        axs[1].legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        # switch on the network train mode
        self.source_nn.train()
        self.target_nn.train()
#         self.bn_process(best_parameters)
        
    def plot_confusion_matrix(self, best_parameters, train_model, dataset, threshold):
        # switch on the network test mode
        if train_model:
            self.train_evaluate(best_parameters, save_model=True)
        self.source_nn.load_state_dict(torch.load(os.path.join(self.model_path, 'Best_Source_NN.pt')))
        self.target_nn.load_state_dict(torch.load(os.path.join(self.model_path, 'Best_Target_NN.pt')))
        self.source_nn.eval()
        self.target_nn.eval()

        domain_idx_dct = {
#             'cl': 0,
            'pdx': 0,
            'tcga': 1
        }
        batGenTest = self.batGenTests[domain_idx_dct[dataset]]
        xdata_test = batGenTest.xdata
        labels_test = batGenTest.labels

        xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
        if best_parameters['use_source'] == 1:
            # ensemble: take average of predicted AADRC and predicted probability
            outputs, target_nn_predictions = self.target_nn(x=xdata_test) 
            predicted_probabilities = target_nn_predictions['Predictions']
            predicted_aadrc, _ = self.source_nn(x=xdata_test) 
            average_predictions = torch.stack([predicted_aadrc, predicted_probabilities]).mean(dim=0)
            predictions = average_predictions.cpu().data.detach().numpy().flatten()
        else:
            outputs, target_nn_predictions = self.target_nn(x=xdata_test) 
            predicted_probabilities = target_nn_predictions['Predictions']
            predictions = predicted_probabilities.cpu().data.detach().numpy().flatten()
    #             print('After Sigmoid', predictions)
        if len(np.unique(labels_test)) == 1:
            print(f"Test set Idx {count} has only response {np.unique(labels_test)}")
            return

        threshold_predictions = np.where(predictions >= threshold, 1, 0)
        print('Distribution of Predictions by Model')
        print(np.quantile(predictions, [0, 0.25,0.5,0.75,1]))
        cm = confusion_matrix(labels_test, threshold_predictions)
        TN, FP, FN, TP = cm.ravel()
        print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
        array_cm = np.array([[TP, FN], [FP, TN]])
        classes = ['1', '0']
        plt.figure(figsize=(3, 3))
        ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False,fmt='.5g')
        ax.set(title="Confusion Matrix", xlabel="Predicted Label", ylabel="Actual Label")
        plt.show()
        
        # switch on the network train mode
        self.source_nn.train()
        self.target_nn.train()
#         self.bn_process(best_parameters)
                
