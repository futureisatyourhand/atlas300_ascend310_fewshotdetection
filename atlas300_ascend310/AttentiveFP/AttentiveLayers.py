# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model
   Description :
   Author :       erik_xiong
   date：          2019-05-28
-------------------------------------------------
   Change Activity:
                   2019-05-28:
-------------------------------------------------
"""
__author__ = 'erik_xiong'

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter_
from torch_scatter import *
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
import os
import time
import datetime

bond_pad_num = 128
atom_max = 32
batch_size = 16

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-06, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x    


class graphAttention(nn.Module):
    def softmax(self, x, index, num=None):
        #print(index)
        #print(scatter_max(x, index, dim=0, dim_size=num)[0][index])
        x = x - scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-8)
        return x

    def __init__(self, bond_dim, fingerprint_dim, p_dropout):
        super(graphAttention, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)
        self.encoder = nn.Sequential(
            LinearBn(bond_dim, fingerprint_dim * fingerprint_dim),
            nn.ReLU(inplace=True),
        )

        self.align = nn.Linear(fingerprint_dim*2,1)
        self.attend = LinearBn(fingerprint_dim, fingerprint_dim)
        self.gru = nn.GRUCell(fingerprint_dim, fingerprint_dim)

    def forward(self, atom, bond_index, bond):
        num_atom, atom_dim = atom.shape
        num_bond, bond_dim = bond.shape
        ###############################################################
        #bond_index = bond_index.int()
        #print("<<<<<",atom.detach().numpy().dtype, bond.detach().numpy().dtype, bond_index.detach().numpy().dtype)
        #merge_data = np.concatenate((atom.detach().numpy().ravel(), bond.detach().numpy().ravel(), bond_index.detach().numpy().ravel()))
        #print (merge_data.shape, merge_data.dtype, merge_data[1], merge_data[-1])
        #np.save('atom_pkg', merge_data.astype(np.float32))
        ###############################################################

        #input：1.bond  2.bond_index  3.atom
        start =time.process_time()
        bond_index = bond_index.t().contiguous()
        neighbor_atom = atom[bond_index[1]]  # 获得所有边的第二个节点的特征
        target_atom = atom[bond_index[0]]  # 获得所有边的第一个节点的特征 

        ###############################################################
        #input：1，bond+bond_index 2，neighbor_atom 3，target_atom 4，atom
        bond = self.encoder(bond).view(-1, atom_dim, atom_dim)  # 将边特征转变为节点特征大小  (12,128,128)
        neighbor = neighbor_atom.view(-1, 1, atom_dim) @ bond  # 12,1,128        
        neighbor = neighbor.view(-1, atom_dim) 

        feature_align = torch.cat([target_atom, neighbor],dim=-1)

        align_score = F.leaky_relu(self.align(self.dropout(feature_align)))
        drop = self.dropout(neighbor)
        atend_neigh = self.attend(drop)

        attention_weight = self.softmax(align_score, bond_index[0], num=num_atom)

        attention =torch.mul(attention_weight, atend_neigh)

        context = scatter_('add', attention, bond_index[0], dim_size=num_atom)
        
        start = time.process_time()
        context = F.elu(context) 
        update = self.gru(context, atom)
        end = time.process_time()
        #print ('time 4: ', (end-start))
        #print ('####:Running stack_model predict time: %s Seconds'%(end2-end1))
        return update

class superatomAttention(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x - scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-8)
        return x
    
    def __init__(self, fingerprint_dim, p_dropout):
        super(superatomAttention, self).__init__()
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.align = nn.Linear(2*fingerprint_dim,1)
        self.attend = LinearBn(fingerprint_dim, fingerprint_dim)

        self.gru = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        
    def forward(self, superatom, atom, mol_index):

        superatom_num = mol_index.max().item() + 1 # number of molecules in a batch

        superatom_expand = superatom[mol_index]

        feature_align = torch.cat([superatom_expand, atom],dim=-1)

        align_score = F.leaky_relu(self.align(self.dropout(feature_align)))

        attention_weight = self.softmax(align_score, mol_index, num=superatom_num)

        context = scatter_('add', torch.mul(attention_weight, self.attend(self.dropout(atom))), \
                           mol_index, dim_size=superatom_num)
        context = F.elu(context)

        update = self.gru(context, superatom) 

        return update, attention_weight
    

class Fingerprint(torch.nn.Module):
    def __init__(self, num_target, fingerprint_dim, K=3, T=3, p_dropout=0.2, atom_dim=39, bond_dim=10):
        super(Fingerprint, self).__init__()
        self.K = K
        self.T = T
        self.dropout = nn.Dropout(p=p_dropout)

        self.preprocess = nn.Sequential(
            LinearBn(atom_dim, fingerprint_dim),
            nn.ReLU(inplace=True),
        )
    
        """self.preprocess = nn.Sequential(
            LinearBn(atom_dim, fingerprint_dim),
            nn.ReLU(inplace=True),
        )"""
    
        self.propagate = nn.ModuleList([graphAttention(bond_dim, fingerprint_dim, p_dropout) for _ in range(K)])
        self.superGather = nn.ModuleList([superatomAttention(fingerprint_dim, p_dropout) for _ in range(T)])
        # weight for initialize superAtom state 
        self.sum_importance = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.sum_importance.data.fill_(0)
    
        self.predict = nn.Sequential(
            LinearBn(fingerprint_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(512, num_target),
        )


    def extract_params(self, model, prefix):
        if not os.path.exists(prefix):
            os.mkdir(prefix)

        for pm in (model.named_parameters()):
            print(pm[0], pm[1].shape, pm[1].dtype)
            np.save(prefix+pm[0], pm[1].detach().numpy())    

        ir = model.modules()
        bn_cnt = 0
        for m in ir:
            name = str(m)
            isbn = name[:11]
            if isbn=='BatchNorm1d':
                bn_cnt = bn_cnt+1
                #mn, vr, wt, bs, eps = m.running_mean, m.running_var, m.weight, m.bias, m.eps
                #ww = 1/torch.sqrt(vr+m.eps) * wt
                #bb = 0 - mn*ww + bs
                #bn1 = ww*fc + bb 
                #print (1000*bn, 1000*bn1)
                mn, vr, wt, bs, eps = m.running_mean.detach().numpy().astype(np.float64), m.running_var.detach().numpy().astype(np.float64), \
                m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64), m.eps
                ww = (1/np.sqrt(vr+m.eps) * wt).astype(np.float32)
                bb = (0 - mn*ww + bs).astype(np.float32)
                np.save(prefix+'bn.gamma'+str(bn_cnt), ww)    
                np.save(prefix+'bn.beta'+str(bn_cnt), bb)    


    def ams_forward(self, atom, bond, bond_index, mol_index, superatom_num, Sentinel):
        ###############################################
        ### by whb: 
        ### make up a blank to align the bond and atoms
        ###############################################
        start = datetime.datetime.now()
        fill_bonds = [[atom.shape[0], atom.shape[0]+1] for i in range(bond_index.shape[0],bond_pad_num)]
        fill_bonds = np.array(fill_bonds, dtype=np.int64)
        fill_bonds = torch.from_numpy(fill_bonds)

        fill_mol = [superatom_num for i in range(atom.shape[0],atom_max)]
        fill_mol = np.array(fill_mol, dtype=np.int64)
        fill_mol = torch.from_numpy(fill_mol)
        #print ('[AMS]fill_bond shapes: ', fill_bonds.shape, bond_index.shape, atom.shape, fill_mol.shape)

        atom = torch.cat([atom, torch.zeros((atom_max-atom.shape[0],39))], 0)
        bond_index = torch.cat([bond_index, fill_bonds], 0)
        bond = torch.cat([bond, torch.zeros((bond_pad_num-bond.shape[0], 10))], 0)
        mol_index = torch.cat([mol_index, fill_mol])
        #print('[AMS]atom inputs shapes: ', atom.shape, bond_index.shape, bond.shape, mol_index.shape)
	
        mol_index = 1 - mol_index	## special process for AI-accelerater, ignore it
        #print("<<<<<",atom.detach().numpy().shape, bond.detach().numpy().shape, bond_index.detach().numpy().shape)
        package_data = np.concatenate((atom.detach().numpy().ravel(), bond.detach().numpy().ravel(), bond_index.detach().numpy().ravel(), mol_index.detach().numpy().ravel()))

        import ctypes
        #mylib = ctypes.cdll.LoadLibrary("../inferences/InferLib/out/infer.so")
        data = package_data.reshape(1, -1)
        data = np.repeat(data, batch_size, axis=0).reshape(-1)
        data = data.astype(np.float32)
        dataptr = data.ctypes
        #dataptr = (ctypes.c_float*data.shape[0])(*data)
     
        result = np.ones((batch_size), np.float32)
        resultptr = (ctypes.c_float*result.shape[0])(*result)
        end = datetime.datetime.now()
        print("AMS-ACC preprocess Time: ", end-start, 's')
        

        #######################################################
        #const string s1 = "int8";
        #const string s2 = "int16";
        #const string s3 = "int32";
        #const string s4 = "int64";
        #const string s5 = "float32";
        #const string s6 = "float64";

        ###############################################
        ### by whb: 
        ### start using AMS-AI ACC make inference
        ###############################################
        # data, bs, ch, wid, hig, total_len, dtype_size, dtype_sel 
        print ('[AMS]Call AMS-AI ACC Interface... Batch_size = ', batch_size)
        mylib = ctypes.cdll.LoadLibrary("infer.so")
        start = datetime.datetime.now()
        mylib._Z5inferPfiiiiiiiS_i(dataptr, batch_size, 1, 1, 2816, batch_size*2816, 4, 5, resultptr, Sentinel)
        end = datetime.datetime.now()
        result = np.ctypeslib.as_array(resultptr).reshape(batch_size)
        print("[AMS] AMS-ACC Return result :", result.shape)
        print (result)
        print('Batch_size =', batch_size, '\tAMS-ACC use Time: ', end-start, 's')
        return result


    def forward(self, atom, bond, bond_index, mol_index, Sentinel):
        num_atom, atom_dim = atom.shape
        superatom_num = mol_index.max()+1
        superatoms = []

        print('''
###############################################
### package interface from here....
### package (atom, bond, bond_index, mol_index) together
### become one numpy-array with size : (%d, 2816)
###############################################'''%(batch_size))
        ######## 1. use AMS AI-ACC make inference
        predict = self.ams_forward(atom, bond, bond_index, mol_index, superatom_num, Sentinel)

        ######## 2. use pytorch-gpu make inference
        start = datetime.datetime.now()
        tmp = atom.cpu().detach()
        atom = self.preprocess(atom)
        #self.extract_params(self.preprocess, 'parms/preprocess/')

        for k in range(self.K):
            atom = self.propagate[k](atom, bond_index, bond)
            #self.extract_params(self.propagate[k], 'parms/propagate'+str(k)+'/')
        
        superatom = scatter_('add', atom, mol_index, dim_size=superatom_num) # get init superatom by sum

        #print(superatom.shape, atom.shape, mol_index)
        for t in range(self.T):
            #self.extract_params(self.superGather[t], 'parms/superGather'+str(t)+'/')
            superatom, attention_weight = self.superGather[t](superatom, atom, mol_index) 

        predict = self.predict(superatom)
        tmp = predict.cpu().detach().numpy()
        #self.extract_params(self.predict, 'parms/predict/')
        end = datetime.datetime.now()
        print("###############################################")
        print("###############################################")
        print ('Pytorch result: ', tmp.shape)
        print (tmp)
        print ('Batch_size =', 1, '\tGPU use Time:  ', end-start, 's')
        return predict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

    

class Fingerprint_viz(torch.nn.Module):
    def __init__(self, num_target, fingerprint_dim, K=3, T=3, p_dropout=0.2, atom_dim=40, bond_dim=10):
        super(Fingerprint_viz, self).__init__()
        self.K = K
        self.T = T
        self.dropout = nn.Dropout(p=p_dropout)

        self.preprocess = nn.Sequential(
            LinearBn(atom_dim, fingerprint_dim),
            nn.ReLU(inplace=True),
        )
    
        self.propagate = nn.ModuleList([graphAttention(bond_dim, fingerprint_dim, p_dropout) for _ in range(K)])
#         self.superGather = superatomAttention(fingerprint_dim, p_dropout=p_dropout)
        self.superGather = nn.ModuleList([superatomAttention(fingerprint_dim, p_dropout) for _ in range(T)])
    
        self.predict = nn.Sequential(
            LinearBn(fingerprint_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(512, num_target),
        )

    def forward(self, atom, bond, bond_index, mol_index):
        atom = self.preprocess(atom)
        num_atom, atom_dim = atom.shape

        atoms = []
        atoms.append(atom)
        for k in range(self.K):
            atom = self.propagate[k](atom, bond_index, bond)
            atoms.append(atom)

#         atom = torch.stack(atoms)
#         atom = torch.mean(atom, dim=0)
        superatom_num = mol_index.max()+1
        superatom = scatter_('add', atom, mol_index, dim_size=superatom_num) # get init superatom by sum
        superatoms = []
        attention_weight_viz = []
        superatoms.append(superatom)
        
        for t in range(self.T):
            superatom, attention_weight = self.superGather[t](superatom, atom, mol_index) 
            attention_weight_viz.append(attention_weight)
            superatoms.append(superatom)

        predict = self.predict(superatom)

        return predict, atoms, superatoms, attention_weight_viz

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

def null_collate(batch):
    batch_size = len(batch)

    atom = []
    bond = []
    bond_index = []
    mol_index = []  
    label = []
    smiles = []
    offset = 0
    for b in range(batch_size):
        graph = batch[b]
        smiles.append(graph.smiles)
        num_atom = len(graph.atom)
        atom.append(graph.atom)
        if graph.bond.size == 0:
            bond = np.zeros((1, 10), np.uint32)
            bond_index = np.zeros((1, 2), np.uint32)
        bond.append(graph.bond)
        bond_index.append(graph.bond_index + offset)
        mol_index.append(np.array([b] * num_atom))
        
        offset += num_atom
        label.append(graph.label)
    atom = torch.from_numpy(np.concatenate(atom)).float()
    bond = torch.from_numpy(np.concatenate(bond)).float()
    bond_index = torch.from_numpy(np.concatenate(bond_index).astype(np.int32)).long()
    mol_index = torch.from_numpy(np.concatenate(mol_index).astype(np.int32)).long()
    label = torch.from_numpy(np.concatenate(label).astype(np.float)).float()
    
    return smiles, atom, bond, bond_index, mol_index, label


class graph_dataset(Dataset):
    def __init__(self, smiles_list, graph_dict):

        self.graph_dict = graph_dict
        self.smiles_list = smiles_list

    def __getitem__(self, x):

        smiles = self.smiles_list[x]

        graph = self.graph_dict[smiles]

        return graph

    def __len__(self):
        return len(self.smiles_list)

class Graph:
    def __init__(self, smiles, atom, bond, bond_index, label):
        self.smiles = smiles
        self.atom = atom
        self.bond = bond
        self.bond_index = bond_index
        self.label = label
        
    def __str__(self):
        return f'graph of {self.smiles}'
    
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        if os.path.exists(file):
            os.remove(file)
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass

def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError
        
if __name__ == '__main__':
    
    smiles_list = ['C1=CC=CC=C1', 'CNC']
    graph_dict = pickle.load(open('test.pkl',"rb"))
    train_loader = DataLoader(graph_dataset(smiles_list, graph_dict), batch_size=2, collate_fn=null_collate)
    net = Fingerprint(2, 32, atom_dim=39, bond_dim=10)
    for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(train_loader):
        _ = net(atom, bond, bond_index, mol_index)
        break

    print('model success!')
