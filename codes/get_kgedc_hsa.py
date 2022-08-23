
# source activate zp2
"""
Usage:
nohup python codes/get_kgedc_hsa.py  >> logs_hsa/log_kgedc_hsa.txt 2>&1 &

"""

import numpy as np
import pandas as pd
import argparse
import os
import time
import torch
import torch.nn as nn
import pickle
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score

from models.PRODeepSyn_datasets import FastTensorDataLoader
from models.PRODeepSyn_utils import save_args, arg_min, conf_inv, calc_stat, save_best_model, find_best_model, random_split_indices
time_str = str(datetime.now().strftime('%y%m%d%H%M'))
from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import Pipeline
from libkge.embedding import TransE, DistMult, ComplEx, TriModel, DistMult_MCL, ComplEx_MCL, TriModel_MCL
from libkge import KgDataset
from libkge.metrics.classification import auc_roc, auc_pr
from libkge.metrics.ranking import precision_at_k, average_precision
from libkge.metrics.classification import auc_pr, auc_roc


parser = argparse.ArgumentParser()
parser.add_argument('--em_size', type=int, default=200, help="the embeding size")
parser.add_argument('--epoch1', type=int, default=1, help="n epoch")
parser.add_argument('--epoch2', type=int, default=1, help="n epoch")
parser.add_argument('--batch', type=int, default=256, help="batch size")
parser.add_argument('--gpu', type=int, default=1, help="cuda device")
parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
parser.add_argument('--suffix', type=str, default='results_kgedc', help="model dir suffix")
parser.add_argument('--hidden', type=int, default= 8192, help="hidden size, [2048, 4096, 8192],")
parser.add_argument('--lr1', type=float, default=0.01, help="learning rate1")
parser.add_argument('--lr2', type=float, default=0.01, help="learning rate2")

#files
OUTPUT_DIR = 'results_hsa/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

args = parser.parse_args()
out_dir = os.path.join(OUTPUT_DIR, '{}'.format(args.suffix))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

log_file = os.path.join(out_dir, 'cv.log')
logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=logging.INFO)

save_args(args, os.path.join(out_dir, 'args.json'))
test_loss_file = os.path.join(out_dir, 'test_loss.pkl')

if torch.cuda.is_available() and (args.gpu is not None):
    gpu_id = args.gpu
else:
    gpu_id = None

###read the dataset
from torch.utils.data import Dataset
def read_map(map_file):
    d = {}
    with open(map_file, 'r') as f:
        f.readline()
        for line in f:
            k, v = line.rstrip().split('\t')
            d[k] = int(v)
    return d

class FastSynergyDataset(Dataset):
    def __init__(self, drug_feat_file, drug_emb, cell_feat_file, cell_emb, synergy_score_file, use_folds, train=True):
        # self.drug2id = read_map(drug2id_file)
        # self.cell2id = read_map(cell2id_file)
        self.drug_feat1 = np.load(drug_feat_file)
        self.drug_feat2 = drug_emb
        self.cell_feat = np.load(cell_feat_file)
        self.cell_feat2 = cell_emb
        self.samples = []
        self.raw_samples = []
        self.train = train
        valid_drugs = set(drugslist)
        valid_cells = set(cellslist)
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        #drug1-drug2-cell
                        sample = [
                            torch.from_numpy(self.drug_feat1[drugslist.index(drug1)]).float(),
                            torch.from_numpy(np.array(self.drug_feat2.loc[drug1])).float(),
                            torch.from_numpy(self.drug_feat1[drugslist.index(drug2)]).float(),
                            torch.from_numpy(np.array(self.drug_feat2.loc[drug2])).float(),
                            torch.from_numpy(self.cell_feat[cellslist.index(cellname)]).float(),
                            torch.from_numpy(np.array(self.cell_feat2.loc[cellname])).float(),
                            torch.FloatTensor([float(score)]),
                        ]
                        self.samples.append(sample)
                        raw_sample = [drugslist.index(drug1), drugslist.index(drug2), cellslist.index(cellname), score]
                        self.raw_samples.append(raw_sample)
                        if train:
                            ###drug2-drug1-cell
                            sample = [
                                torch.from_numpy(self.drug_feat1[drugslist.index(drug2)]).float(),
                                torch.from_numpy(np.array(self.drug_feat2.loc[drug2])).float(),
                                torch.from_numpy(self.drug_feat1[drugslist.index(drug1)]).float(),
                                torch.from_numpy(np.array(self.drug_feat2.loc[drug1])).float(),
                                torch.from_numpy(self.cell_feat[cellslist.index(cellname)]).float(),
                                torch.from_numpy(np.array(self.cell_feat2.loc[cellname])).float(),
                                torch.FloatTensor([float(score)]),
                            ]
                            self.samples.append(sample)
                            raw_sample = [drugslist.index(drug2), drugslist.index(drug1), cellslist.index(cellname), score]
                            self.raw_samples.append(raw_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def drug_feat1_len(self):
        return self.drug_feat1.shape[-1]

    def drug_feat2_len(self):
        return self.drug_feat2.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]
    def cell_feat2_len(self):
        return self.cell_feat2.shape[-1]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1_f1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d1_f2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        d2_f1 = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        d2_f2 = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][4], 0) for i in indices], dim=0)
        c2 = torch.cat([torch.unsqueeze(self.samples[i][5], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][6], 0) for i in indices], dim=0)
        return d1_f1, d1_f2, d2_f1, d2_f2, c,c2, y


##create model
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, drug_feat1_len:int,  drug_feat2_len:int, cell_feat_len:int, cell_feat2_len:int, hidden_size: int):
        super(DNN, self).__init__()

        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat1_len*2),
            nn.Linear(drug_feat1_len*2, drug_feat1_len),
        )

        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat2_len*2),
            nn.Linear(drug_feat2_len*2, drug_feat2_len),
        )

        # 768 / 2 = 384
        self.cell_network = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat_len ),
            nn.Linear(cell_feat_len, 384),
        )

        self.cell_network2 = nn.Sequential(
            nn.Linear(cell_feat2_len, cell_feat2_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat2_len ),
            nn.Linear(cell_feat2_len, 384),
        )

        self.fc_network = nn.Sequential(
            nn.BatchNorm1d(2*(drug_feat1_len + drug_feat2_len)+ 768),
            nn.Linear(2*(drug_feat1_len + drug_feat2_len)+ 768, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor, cell_feat: torch.Tensor, cell_feat2: torch.Tensor):
        drug1_feat1_vector = self.drug_network1( drug1_feat1 ) 
        drug1_feat2_vector = self.drug_network2( drug1_feat2 )
        drug2_feat1_vector = self.drug_network1( drug2_feat1 ) 
        drug2_feat2_vector = self.drug_network2( drug2_feat2 )
        cell_feat_vector = self.cell_network(cell_feat)
        cell_feat_vector2 = self.cell_network2(cell_feat2)

        # cell_feat_vector = cell_feat
        feat = torch.cat([drug1_feat1_vector, drug1_feat2_vector, drug2_feat1_vector, drug2_feat2_vector, cell_feat_vector, cell_feat_vector2], 1)
        out = self.fc_network(feat)
        return out

class DNN_orig(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DNN_orig, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor, cell_feat: torch.Tensor):
        feat = torch.cat([drug1_feat1, drug1_feat2, drug2_feat1, drug2_feat2, cell_feat], 1)
        out = self.network(feat)
        return out

#useful functions
def create_model(data, hidden_size, gpu_id=None):
    # model = DNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    model = DNN(data.drug_feat1_len(), data.drug_feat2_len(), data.cell_feat_len(), data.cell_feat2_len(), hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model

def step_batch(model, batch, loss_func, gpu_id=None, train=True):
    if gpu_id is not None:
        batch = [x.cuda(gpu_id) for x in batch]
    drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2, y_true = batch
    # if gpu_id is not None:
        # drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = drug1_feats1.cuda(gpu_id), drug1_feats2.cuda(gpu_id),drug1_feats3.cuda(gpu_id), drug2_feats1.cuda(gpu_id), drug2_feats2.cuda(gpu_id), drug2_feats3.cuda(gpu_id), cell_feats.cuda(gpu_id), y_true.cuda(gpu_id)
    if train:
        y_pred = model(drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2)
    else:
        yp1 = model(drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2)
        yp2 = model(drug2_feats1, drug2_feats2, drug1_feats1, drug1_feats2, cell_feats, cell_feats2)
        y_pred = (yp1 + yp2) / 2
    loss = loss_func(y_pred, y_true)
    return loss


def train_epoch(model, loader, loss_func, optimizer, gpu_id=None):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = step_batch(model, batch, loss_func, gpu_id)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def eval_epoch(model, loader, loss_func, gpu_id=None):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch in loader:
            loss = step_batch(model, batch, loss_func, gpu_id, train=False)
            epoch_loss += loss.item()
    return epoch_loss


def train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, gpu_id,
                sl=False, mdl_dir=None):
    min_loss = float('inf')
    angry = 0
    for epoch in range(1, n_epoch + 1):
        trn_loss = train_epoch(model, train_loader, loss_func, optimizer, gpu_id)
        trn_loss /= train_loader.dataset_len
        val_loss = eval_epoch(model, valid_loader, loss_func, gpu_id)
        val_loss /= valid_loader.dataset_len
        if val_loss < min_loss:
            angry = 0
            min_loss = val_loss
            if sl:
                save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)
        else:
            angry += 1
            if angry >= patience:
                break
    if sl:
        model.load_state_dict(torch.load(find_best_model(mdl_dir)))
    return min_loss


def eval_model(model, optimizer, loss_func, train_data, test_data,
               batch_size, n_epoch, patience, gpu_id, mdl_dir):
    tr_indices, es_indices = random_split_indices(len(train_data), test_rate=0.1)
    train_loader = FastTensorDataLoader(*train_data.tensor_samples(tr_indices), batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(*train_data.tensor_samples(es_indices), batch_size=len(es_indices) // 4)
    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data) // 4)
    train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, gpu_id,
                sl=True, mdl_dir=mdl_dir)
    test_loss = eval_epoch(model, test_loader, loss_func, gpu_id)
    test_loss /= len(test_data)
    return test_loss

SYNERGY_FILE = 'data/merged_synergy_hsa_folds.txt'
DRUG_FEAT_FILE = 'data/merged_drug_inform_feat.npy'
CELL_FEAT_FILE = 'data/merged_cell_feat.npy'

import numpy as np
import pandas as pd
data = pd.read_csv(SYNERGY_FILE, sep='\t', header=0)
data.columns = ['drugname1','drugname2','cell_line','synergy','fold']
drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2'])))) #38
drugscount = len(drugslist)
cellslist = sorted(list(set(data['cell_line']))) 
cellscount = len(cellslist)

threshold = 3.87

n_folds = 1
n_delimiter = 60
test_losses = []
test_pccs = []
class_stats = np.zeros((n_folds, 7))
# for test_fold in range(n_folds):
test_fold = 0
valid_fold = list(range(10))[test_fold-1]
train_fold = [ x for x in list(range(10)) if x != test_fold and x != valid_fold ]
print(train_fold, valid_fold, test_fold)

test_data = data[data['fold']== test_fold]
valid_data = data[data['fold']==valid_fold]
train_data = data[(data['fold']!=test_fold) & (data['fold']!=valid_fold) ]
print('processing test fold {0} train folds {1} valid folds{2}.'.format(test_fold, train_fold, valid_fold))
print('test shape{0} train shape{1} valid shape {2}'.format(test_data.shape, train_data.shape, valid_data.shape))


##kge embedding module
kgdata = []
for cellname in cellslist:
    # cellname = cellslist[0]
    eachdata = data[data['cell_line']==cellname]
    # net1_data =  eachdata[eachdata['synergy']>=10]
    # print(eachdata.shape, net1_data.shape)
    for each in eachdata.values:
        drugname1, drugname2, cell_line, synergy, fold = each
        if float(synergy) >= threshold:
            kgdata.append([drugname1, cell_line, drugname2])
            kgdata.append([drugname2, cell_line, drugname1])

kgdata = np.array(kgdata)
dataset = KgDataset(name='kgdds')
dataset.load_triples(kgdata, tag='train')

train_data = dataset.data['train']
nb_entities = dataset.get_ents_count()
nb_relations = dataset.get_rels_count()

print("Initializing the knowledge graph embedding model... ")
seed = 123
# model = TriModel(seed=seed, verbose=2)
# model = TransE(seed=seed, verbose=2)
model = DistMult(seed=seed, verbose=2)
# model = ComplEx(seed=seed, verbose=2)


# set model parameters
model_params = {
    'em_size': args.em_size,
    'lr': args.lr1,
    'optimiser': "AMSgrad",
    'log_interval': 10,
    'nb_epochs': args.epoch1,
    'nb_negs': 8,
    'batch_size': 1000,
    'initialiser': 'xavier_uniform',
    'nb_ents': nb_entities,
    'nb_rels': nb_relations
}


model.set_params(**model_params)

print("Training ... ")
model.fit(X=train_data, y=None)

embeddings =  model.get_embeddings()
embeddings_ents = embeddings['ents']
embeddings_rels = embeddings['rels']

entsid2name = {v:k for k,v in dataset.ent_mappings.items() }
entsnames = list(dataset.ent_mappings.keys())
relsid2name = {v:k for k,v in dataset.rel_mappings.items() }
relsnames = list(dataset.rel_mappings.keys())

embeddings_ents = pd.DataFrame(embeddings_ents[:nb_entities])
embeddings_ents.index = entsnames

embeddings_rels = pd.DataFrame(embeddings_rels[:nb_relations])
embeddings_rels.index = relsnames

drugslist_noemb = [ x for x in drugslist if x not in entsnames ]
embeddings_zeros = pd.DataFrame([[0]*embeddings_ents.shape[1]] * len(drugslist_noemb))
embeddings_zeros.index = drugslist_noemb
embeddings_ents = pd.concat([embeddings_ents, embeddings_zeros])
pd.DataFrame(embeddings_ents).to_csv(out_dir+ '/embeddings_ents.txt',sep='\t', header=None, index=True)
pd.DataFrame(embeddings_rels).to_csv(out_dir + '/embeddings_rels.txt',sep='\t', header=None, index=True)


####predictor module
####predictor module
####predictor module
print('begining predictor......')
mdl_dir = os.path.join(out_dir, str(test_fold))
if not os.path.exists(mdl_dir):
    os.makedirs(mdl_dir)

#embeddings
embeddings_drug = pd.read_csv(out_dir + '/embeddings_ents.txt',sep='\t', header=None, index_col=0)
embeddings_cell = pd.read_csv(out_dir + '/embeddings_rels.txt',sep='\t', header=None, index_col=0)

logging.info("Outer: train folds {}, valid folds {} ,test folds {}".format(train_fold, valid_fold, test_fold))
logging.info("-" * n_delimiter)

best_hs, best_lr = args.hidden, args.lr2
logging.info("Best hidden size: {} | Best learning rate: {}".format(best_hs, best_lr))

##preprocess data
##preprocess data
train_data = FastSynergyDataset( DRUG_FEAT_FILE, embeddings_drug, CELL_FEAT_FILE, embeddings_cell, SYNERGY_FILE, use_folds=train_fold)
valid_data = FastSynergyDataset( DRUG_FEAT_FILE, embeddings_drug, CELL_FEAT_FILE, embeddings_cell,SYNERGY_FILE, use_folds=[valid_fold], train=False)
test_data = FastSynergyDataset( DRUG_FEAT_FILE, embeddings_drug, CELL_FEAT_FILE, embeddings_cell, SYNERGY_FILE, use_folds=[test_fold], train=False)

train_loader = FastTensorDataLoader(*train_data.tensor_samples(), batch_size=args.batch, shuffle=True)
valid_loader = FastTensorDataLoader(*valid_data.tensor_samples(), batch_size=len(valid_data))
test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))

model = create_model(train_data, best_hs, gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
loss_func = nn.MSELoss(reduction='sum')

##train
#train
min_loss = float('inf')
for epoch in range(1, args.epoch2 + 1):
    trn_loss = train_epoch(model, train_loader, loss_func, optimizer, gpu_id)
    trn_loss /= train_loader.dataset_len
    val_loss = eval_epoch(model, valid_loader, loss_func, gpu_id)
    val_loss /= valid_loader.dataset_len
    if epoch % 100 == 0: 
        print("epoch: {} | train loss: {} valid loss {}".format(epoch, trn_loss, val_loss))
    if val_loss < min_loss:
        min_loss = val_loss
        save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)

model.load_state_dict(torch.load(find_best_model(mdl_dir)))

##test predict
##test predict
##test predict
with torch.no_grad():
    for test_each in test_loader:
        test_each = [x.cuda(gpu_id) for x in test_each]
        drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats,cell_feats2, y_true = test_each
        yp1 = model(drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2)
        yp2 = model(drug2_feats1, drug2_feats2, drug1_feats1, drug1_feats2, cell_feats, cell_feats2)
        y_pred = (yp1 + yp2) / 2
        test_loss = loss_func(y_pred, y_true).item()
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_true.cpu().numpy().flatten()
        test_pcc = np.corrcoef(y_pred, y_true)[0, 1]
        test_loss /= len(y_true)
        y_pred_binary = [ 1 if x >= threshold else 0 for x in y_pred ]
        y_true_binary = [ 1 if x >= threshold else 0 for x in y_true ]
        roc_score = roc_auc_score(y_true_binary, y_pred)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
        auprc_score = auc(recall, precision)
        accuracy = accuracy_score( y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary)
        kappa = cohen_kappa_score(y_true_binary, y_pred_binary)

class_stat = [roc_score, auprc_score, accuracy, f1, precision, recall, kappa]
class_stats[test_fold] = class_stat
test_losses.append(test_loss)
test_pccs.append(test_pcc)
logging.info("Test loss: {:.4f}".format(test_loss))
logging.info("Test pcc: {:.4f}".format(test_pcc))
logging.info("*" * n_delimiter + '\n')

##cal the stats in each cell line mse
from sklearn.metrics import mean_squared_error

all_data = pd.read_csv(SYNERGY_FILE, sep='\t', header=0)
test_data_orig = all_data[all_data['fold']==test_fold]
test_data_orig['pred'] = y_pred
test_data_orig.to_csv(out_dir + '/test_data_' + str(test_fold) + '.txt' ,sep='\t',header=True, index=False)
cells_stats = np.zeros((cellscount, 9))
for cellidx in range(cellscount):
    # cellidx = 0
    cellname = cellslist[cellidx]
    each_data = test_data_orig[test_data_orig['cell_line']== cellname ]
    each_true = each_data['synergy'].tolist()
    each_pred = each_data['pred'].tolist()
    each_loss = mean_squared_error(each_true, each_pred)
    each_pcc = np.corrcoef(each_pred, each_true)[0, 1]
    #class
    each_pred_binary = [ 1 if x >= threshold else 0 for x in each_pred ]
    each_true_binary = [ 1 if x >= threshold else 0 for x in each_true ]
    roc_score_each = roc_auc_score(each_true_binary, each_pred)
    precision, recall, _ = precision_recall_curve(each_true_binary, each_pred_binary)
    auprc_score_each = auc(recall, precision)
    accuracy_each = accuracy_score(each_true_binary, each_pred_binary)
    f1_each = f1_score(each_true_binary, each_pred_binary)
    precision_each = precision_score(each_true_binary, each_pred_binary, zero_division=0)
    recall_each = recall_score(each_true_binary, each_pred_binary)
    kappa_each = cohen_kappa_score(each_true_binary, each_pred_binary)
    t = [each_loss, each_pcc, roc_score_each, auprc_score_each, accuracy_each, f1_each, precision_each, recall_each, kappa_each]
    cells_stats[cellidx] = t

pd.DataFrame(cells_stats).to_csv(out_dir+'/test_data_cells_stats_'+str(test_fold)+'.txt', sep='\t', header=None, index=None)

logging.info("CV completed")
with open(test_loss_file, 'wb') as f:
    pickle.dump(test_losses, f)
mu, sigma = calc_stat(test_losses)
logging.info("MSE: {:.4f} ± {:.4f}".format(mu, sigma))
lo, hi = conf_inv(mu, sigma, len(test_losses))
logging.info("Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi))
rmse_loss = [x ** 0.5 for x in test_losses]
mu, sigma = calc_stat(rmse_loss)
logging.info("RMSE: {:.4f} ± {:.4f}".format(mu, sigma))
pcc_mean, pcc_std = calc_stat(test_pccs)
logging.info("pcc: {:.4f} ± {:.4f}".format(pcc_mean, pcc_std))

reg_stats = pd.DataFrame()
reg_stats['mse'] = test_losses
reg_stats['pcc'] = test_pccs
# reg_stats.loc['stats'] = [str(round(mu,2)) +'±' str(round(sigma,2)), ]
reg_stats.to_csv(out_dir + '/reg_stats.txt', sep='\t', header=None, index=None)

class_stats = np.concatenate([class_stats, class_stats.mean(axis=0, keepdims=True), class_stats.std(axis=0, keepdims=True)], axis=0)
pd.DataFrame(class_stats).to_csv(out_dir + '/class_stats.txt', sep='\t', header=None, index=None)


