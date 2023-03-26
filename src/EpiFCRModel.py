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

from EpiDataHelpers import (
    get_cl_threshold,
    get_cl,
    get_pdx,
    get_tcga,
    get_cl_threshold_cnv,
    get_cl_cnv,
    get_pdx_cnv,
    get_tcga_cnv,
    get_cl_threshold_mut_cnv,
    get_cl_mut_cnv,
    get_pdx_mut_cnv,
    get_tcga_mut_cnv,
    get_dataset
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

        self.feature1 = FeatModule(flags, X)
        self.feature2 = FeatModule(flags, X)
#         self.feature3 = FeatModule(flags, X)

        self.features = [self.feature1, self.feature2]

        self.classifier1 = ClassifierModule(flags)
        self.classifier2 = ClassifierModule(flags)
#         self.classifier3 = ClassifierModule()

        self.classifiers = [self.classifier1, self.classifier2]

    def bn_eval(self):
        for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

    def forward(self, x, domain): # domain is indexed 
        net = self.features[domain](x)
        net, end_points = self.classifiers[domain](net)
        return net, end_points


class DomainAGG(nn.Module):
    def __init__(self, flags, X):
        super(DomainAGG, self).__init__()

        self.feature = FeatModule(flags, X)
        self.classifier = ClassifierModule(flags)
        self.classifierrand = ClassifierModule(flags)

    def bn_eval(self):
        for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

    def forward(self, x, agg_only=True):
        net = self.feature(x)
        net_rand = None
        if agg_only:
            net_agg, end_points = self.classifier(net)
        else:
            net_agg, end_points = self.classifier(net)
            net_rand, _ = self.classifierrand(net)
        return net_agg, net_rand, end_points
    
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
    
class ModelEpiFCR():
    def __init__(self, drug_name, input_name):
        self.drug_name = drug_name
        self.input_name = input_name
        self.model_path = f'./epi_models/{input_name}/{drug_name}/tuned/'
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.logs = f'./epi_logs/{input_name}/{drug_name}/tuned/'
        if not os.path.isdir(self.logs):
            os.makedirs(self.logs)
        self.prediction_path = f'./epi_predictions/{input_name}/{drug_name}/tuned/'
        if not os.path.isdir(self.prediction_path):
            os.makedirs(self.prediction_path)
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def configure(self, flags):
#         for name, para in self.ds_nn.named_parameters():
#             print(name, para.size())
#         for name, para in self.agg_nn.named_parameters():
#             print(name, para.size())

        self.optimizer_ds_nn = optim.SGD(params=[{'params': self.ds_nn.parameters()}],
                                   lr=flags['lr'],
                                   weight_decay=flags['weight_decay'],
                                   momentum=flags['momentum'])

        self.scheduler_ds_nn = lr_scheduler.StepLR(optimizer=self.optimizer_ds_nn, step_size=flags['step_size'],
                                                   gamma=0.1)

        self.optimizer_agg = optim.SGD(params=[{'params': self.agg_nn.feature.parameters()},
                                             {'params': self.agg_nn.classifier.parameters()}],
                                 lr=flags['lr'],
                                 weight_decay=flags['weight_decay'],
                                 momentum=flags['momentum'])

        self.scheduler_agg = lr_scheduler.StepLR(optimizer=self.optimizer_agg,
                                                 step_size=flags['step_size'], gamma=0.1)

        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device) # inputs do not need to go through sigmoid prior to input, like how crossentropyloss does not need to be softmax activated

    def setup(self, flags):    
        # fix all seeds
        random.seed(flags['seed'])
        np.random.seed(flags['seed'])
        torch.manual_seed(flags['seed'])
        torch.cuda.manual_seed_all(flags['seed'])
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        if self.input_name == 'mut_cnv':
            self.ds_nn = DomainSpecificNN(flags, 648).to(self.device)
            self.agg_nn = DomainAGG(flags, 648).to(self.device)
        else:
            self.ds_nn = DomainSpecificNN(flags, 324).to(self.device)
            self.agg_nn = DomainAGG(flags, 324).to(self.device)

        print(self.ds_nn)
        print(self.agg_nn)
        print('flags:', flags)

        flags_log = os.path.join(self.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.configure(flags)

        self.loss_weight_epic = flags['loss_weight_epic']
        self.loss_weight_epif = flags['loss_weight_epif']
        self.loss_weight_epir = flags['loss_weight_epir'] # how much weight given to random classifier in objective loss

        self.best_auroc_val = float('-inf')

    def setup_path(self, flags):        
        drug_name = self.drug_name
        
        self.train_paths = ['cl',
#                           'pdx',
                          'tcga']
        
        self.val_paths = [
#                         'cl',
#                           'pdx',
                          'tcga']
        
        # get test data for particular drug
        self.test_paths = ['cl',
                          'pdx',
                          'tcga']
        
        temp_paths = self.train_paths.copy()
        self.train_paths = []
        print(f'-------{drug_name}-------')
        for t in temp_paths:
            train_df = get_dataset(t, drug_name, 'train', flags, self.input_name)
            self.train_paths.append(train_df)
            print(f'{t} train set size: {len(train_df)}')
            
        temp_paths = self.val_paths.copy()
        self.val_paths = []
        for t in temp_paths:
            val_df = get_dataset(t, drug_name, 'val', flags, self.input_name)
            self.val_paths.append(val_df)
            print(f'{t} validation set size: {len(val_df)}')
            
        temp_paths = self.test_paths.copy()
        self.test_paths = []
        for t in temp_paths:
            test_df = get_dataset(t, drug_name, 'test', flags, self.input_name)
            self.test_paths.append(test_df)
            print(f'{t} test set size: {len(test_df)}')
        
        # for training domain specific 
        self.batGenTrains = []
        for train_path in self.train_paths: 
            # each domain dataset has a BatchImageGenerator object, call get_images_labels_batch to get batch
            batGenTrain = BatchGenerator(flags=flags, file_path=train_path, stage='train')
            self.batGenTrains.append(batGenTrain)
        
        # for training domain agnostic
        self.batGenTrainsDg = []
        for train_path in self.train_paths:
            batGenTrain = BatchGenerator(flags=flags, file_path=train_path, stage='train')
            self.batGenTrainsDg.append(batGenTrain)

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
#         self.batGenTest = BatchGenerator(flags=flags, file_path=self.unseen_data_path, stage='test')

        self.candidates = np.arange(0, len(self.batGenTrains))
        
    
    def bn_process(self, flags):
        if flags['bn_eval'] == 1:
            self.ds_nn.bn_eval()
            self.agg_nn.bn_eval()

    def train_agg_nn(self, ite, flags): # ite = num_iter in training loop
        # set domain specific to eval
        self.ds_nn.eval()
        self.agg_nn.train()
        self.bn_process(flags)
        
        # index out 2 domain specific modules (not the same)
        candidates = list(self.candidates)
        index_val = np.random.choice(candidates, size=1)[0]
        candidates.remove(index_val)
        index_trn = np.random.choice(candidates, size=1)[0]
        assert index_trn != index_val

        # get the inputs and labels from the data reader
        agg_ent_loss = 0.0
        epir_loss = 0.0
        epic_loss = 0.0
        epif_loss = 0.0
        for index in range(len(self.batGenTrainsDg)):
            xdata_train, labels_train = self.batGenTrainsDg[index].get_xdata_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(xdata_train, dtype=np.float32)), torch.from_numpy(
                np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs, requires_grad=False).to(self.device), \
                             Variable(labels, requires_grad=False).long().to(self.device)

            # forward
            outputs_agg, outputs_rand, _ = self.agg_nn(x=inputs, agg_only=False)

            # loss
            agg_ent_loss += self.loss_fn(outputs_agg.squeeze(), labels.float())
            epir_loss += self.loss_fn(outputs_rand.squeeze(), labels.float()) * self.loss_weight_epir

            if index == index_val:
                assert index != index_trn
                # train domain-agnostic classifier by taking in output from domain-specific feature extractor
                if ite >= flags['ite_train_epi_c']:
                    net = self.ds_nn.features[index_trn](inputs)
                    outputs_val, _ = self.agg_nn.classifier(net)
                    epic_loss += self.loss_fn(outputs_val.squeeze(), labels.float()) * self.loss_weight_epic
                
                # train domain-agnostic feature extractor by passing output to domain-specific feature extractor
                if ite >= flags['ite_train_epi_f']:
                    net = self.agg_nn.feature(inputs)
                    outputs_val, _ = self.ds_nn.classifiers[index_trn](net)
                    epif_loss += self.loss_fn(outputs_val.squeeze(), labels.float()) * self.loss_weight_epif

        # init the grad to zeros first
        self.optimizer_agg.zero_grad()

        # backward your network
        (agg_ent_loss + epir_loss + epic_loss + epif_loss).backward()

        # optimize the parameters
        self.optimizer_agg.step()
        
        # change order due to pytorch ver
        self.scheduler_agg.step()

        flags_log = os.path.join(self.logs, 'agg_ent_loss.txt')
        write_log(str(agg_ent_loss.item()), flags_log)
        flags_log = os.path.join(self.logs, 'epir_loss.txt')
        write_log(str(epir_loss.item()), flags_log)
        if ite >= flags['ite_train_epi_c']:
            flags_log = os.path.join(self.logs, 'epic_loss.txt')
            write_log(str(epic_loss.item()), flags_log)
        if ite >= flags['ite_train_epi_f']:
            flags_log = os.path.join(self.logs, 'epif_loss.txt')
            write_log(str(epif_loss.item()), flags_log)

    def train_agg_nn_warm(self, ite, flags):
        # warm up domain agnostic module, without epi-r 
        self.agg_nn.train()
        self.bn_process(flags)

        # get the inputs and labels from the data reader
        agg_ent_loss = 0.0
        for index in range(len(self.batGenTrainsDg)):
            xdata_train, labels_train = self.batGenTrainsDg[index].get_xdata_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(xdata_train, dtype=np.float32)), torch.from_numpy(
                np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs, requires_grad=False).to(self.device), \
                             Variable(labels, requires_grad=False).long().to(self.device)

            # forward
            outputs, _, _ = self.agg_nn(x=inputs)

            # loss
            agg_ent_loss += self.loss_fn(outputs.squeeze(), labels.float())

        # init the grad to zeros first
        self.optimizer_agg.zero_grad()

        # backward your network
        agg_ent_loss.backward()

        # optimize the parameters
        self.optimizer_agg.step()
        
        # change order due to pytorch ver
        self.scheduler_agg.step()

        flags_log = os.path.join(self.logs, 'agg_ent_loss.txt')
        write_log(str(agg_ent_loss.item()), flags_log)

    def train_ds_nn(self, ite, flags):
        # train domain specific modules
        self.ds_nn.train()
        self.bn_process(flags)

        # get the inputs and labels from the data reader
        ent_loss = 0.0
        for index in range(len(self.batGenTrains)): # for each domain
            xdata_train, labels_train = self.batGenTrains[index].get_xdata_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(xdata_train, dtype=np.float32)), torch.from_numpy(
                np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs, requires_grad=False).to(self.device), \
                             Variable(labels, requires_grad=False).long().to(self.device)

            # forward
            outputs, _ = self.ds_nn(x=inputs, domain=index) # 1 index -> 1 dataset -> 1 domain 

            # loss
            loss = self.loss_fn(outputs.squeeze(), labels.float())

            ent_loss += loss

        self.optimizer_ds_nn.zero_grad()

        # backward your network
        ent_loss.backward()

        # optimize the parameters
        self.optimizer_ds_nn.step()
        
        # change order due to pytorch ver
        self.scheduler_ds_nn.step()

        flags_log = os.path.join(self.logs, 'ds_nn_loss_log.txt')
        write_log(str(ent_loss.item()), flags_log)

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

    def test(self, flags, ite, log_prefix='', log_dir='epi_logs/', batGenTest=None):

        # switch on the network test mode
        self.agg_nn.eval()

        xdata_test = batGenTest.xdata
        labels_test = batGenTest.labels

        xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
        tuples = self.agg_nn(xdata_test, agg_only=True)

        predictions = tuples[-1]['Predictions']
        if len(np.unique(labels_test)) == 1:
            roc_auc = np.nan
        else:
            roc_auc = roc_auc_score(labels_test, predictions.cpu().data.detach().numpy())
#         rounded_predictions = torch.round(predictions).cpu().data.detach().numpy()
#         acc = accuracy_score(y_true=labels_test, y_pred=rounded_predictions)
#         print('----------AUROC test----------:', roc_auc)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, AUROC:{}\n'.format(ite, roc_auc))
        f.close()

        # switch on the network train mode
        self.agg_nn.train()
        self.bn_process(flags)

        return roc_auc

    def train_evaluate(self, flags, save_model=False):
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)
        
        self.ds_nn.train()
        self.agg_nn.train()
        self.bn_process(flags) # set to eval all batch norms
        
        max_mean_auroc = float('-inf')
        for ite in range(flags['loops_train']):
            if ite <= flags['loops_warm']:
                self.train_ds_nn(ite, flags)
                
                if (flags['warm_up_agg'] == 1) and (ite <= flags['loops_agg_warm']):
                    self.train_agg_nn_warm(ite, flags)

                if (ite % flags['test_every'] == 0) and (ite != 0) and (flags['warm_up_agg'] != 1):
                    mean_auroc = self.test_workflow(self.batGenVals, flags, ite, prefix='')
            else:
                self.train_ds_nn(ite, flags)
                self.train_agg_nn(ite, flags)

#                 if (ite % flags['test_every'] == 0) and (ite != 0):
                mean_auroc = self.test_workflow(self.batGenVals, flags, ite, prefix='')
                if mean_auroc > max_mean_auroc:
                    max_mean_auroc = mean_auroc
                    if save_model:
                        torch.save(self.agg_nn.state_dict(), os.path.join(self.model_path, 'Best_Model.pt'))

        dct = {'auc': max_mean_auroc}
        print(f'Highest mean AUROC achieved during training: {max_mean_auroc}')
        return dct
    
    def tune_model(self, parameters, total_trials):
        best_parameters, values, experiment, model = optimize(
            experiment_name="episodic_training",
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
        self.agg_nn.load_state_dict(torch.load(os.path.join(self.model_path, 'Best_Model.pt')))
        self.agg_nn.eval()
        
        domain_names_dct = {
            0: 'cell-line',
            1: 'PDX',
            2: 'TCGA'
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
            tuples = self.agg_nn(xdata_test, agg_only=True)

            predictions = tuples[-1]['Predictions']
            raw_outputs = tuples[0].cpu().data.detach().numpy()
            predictions = predictions.cpu().data.detach().numpy().flatten()
            
            np.savetxt(os.path.join(self.prediction_path, f'{domain_names_dct[count]}_testset_groundtruth.csv'), labels_test)
            np.savetxt(os.path.join(self.prediction_path, f'{domain_names_dct[count]}_testset_predictions.csv'), predictions)
            
#             print('After Sigmoid', predictions)
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
        self.agg_nn.train()
        self.bn_process(best_parameters)
        return results_df
    
    def get_train_set_results(self, best_parameters, train_model):
        # switch on the network test mode
        if train_model:
            self.train_evaluate(best_parameters, save_model=True)
        self.agg_nn.load_state_dict(torch.load(os.path.join(self.model_path, 'Best_Model.pt')))
        self.agg_nn.eval()
        
        # get test data for particular drug
        train_all_paths = ['cl',
                          'pdx',
                          'tcga']
        train_all_datasets = []
        for t in train_all_paths:
            train_all_datasets.append(get_dataset(t, self.drug_name, 'train_all', best_parameters, self.input_name))
        
        domain_names_dct = {
            0: 'cell-line',
            1: 'PDX',
            2: 'TCGA'
        }
        results_df = pd.DataFrame()
        for count, batGenTest in enumerate(train_all_datasets):
            xdata_test = batGenTest.drop("response", axis = 1).values
            labels_test = batGenTest.iloc[:]["response"].values
            response_value_counts = pd.value_counts(labels_test).to_dict()

            xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
            tuples = self.agg_nn(xdata_test, agg_only=True)

            predictions = tuples[-1]['Predictions']
            raw_outputs = tuples[0].cpu().data.detach().numpy()
            predictions = predictions.cpu().data.detach().numpy().flatten()
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
        self.agg_nn.train()
        self.bn_process(best_parameters)
        return results_df
    
    def get_roc_pr_curves(self, best_parameters, train_model, dataset):
        # switch on the network test mode
        if train_model:
            self.train_evaluate(best_parameters, save_model=True)
        self.agg_nn.load_state_dict(torch.load(os.path.join(self.model_path, 'Best_Model.pt')))
        self.agg_nn.eval()

        domain_idx_dct = {
            'cl': 0,
            'pdx': 1,
            'tcga': 2
        }
        batGenTest = self.batGenTests[domain_idx_dct[dataset]]
        xdata_test = batGenTest.xdata
        labels_test = batGenTest.labels

        xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
        tuples = self.agg_nn(xdata_test, agg_only=True)

        predictions = tuples[-1]['Predictions']
        raw_outputs = tuples[0].cpu().data.detach().numpy()
        predictions = predictions.cpu().data.detach().numpy().flatten()
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
        self.agg_nn.train()
        self.bn_process(best_parameters)
        
    def plot_confusion_matrix(self, best_parameters, train_model, dataset, threshold):
        # switch on the network test mode
        if train_model:
            self.train_evaluate(best_parameters, save_model=True)
        self.agg_nn.load_state_dict(torch.load(os.path.join(self.model_path, 'Best_Model.pt')))
        self.agg_nn.eval()

        domain_idx_dct = {
            'cl': 0,
            'pdx': 1,
            'tcga': 2
        }
        batGenTest = self.batGenTests[domain_idx_dct[dataset]]
        xdata_test = batGenTest.xdata
        labels_test = batGenTest.labels

        xdata_test = Variable(torch.from_numpy(np.array(xdata_test, dtype=np.float32))).to(self.device)
        tuples = self.agg_nn(xdata_test, agg_only=True)

        predictions = tuples[-1]['Predictions']
        raw_outputs = tuples[0].cpu().data.detach().numpy()
        predictions = predictions.cpu().data.detach().numpy().flatten()
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
        self.agg_nn.train()
        self.bn_process(best_parameters)