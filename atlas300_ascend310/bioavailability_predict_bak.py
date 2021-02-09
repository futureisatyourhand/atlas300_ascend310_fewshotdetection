import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('smiles', type=str, help='input smiles')

args = parser.parse_args()

# In[1]:


import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
import random
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
from sklearn.model_selection import KFold

#then import my own modules
from timeit import default_timer as timer
from AttentiveFP.featurizing import graph_dict
from AttentiveFP.AttentiveLayers import Fingerprint, graph_dataset, null_collate, Graph, Logger, time_to_str


# In[2]:


#cuda_aviable = torch.cuda.is_available()
device = torch.device('cpu')


# In[3]:


from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns; sns.set()
from IPython.display import SVG, display
import time
# import sascorer


# In[4]:

#print ("0")
#start =time.process_time()

smilesList = []
smilesList.append(args.smiles)
graph_dict = graph_dict(smilesList)

#end =time.process_time()
#print ('1:Running time: %s Seconds'%(end-start))

# In[6]:

#from pytorch2caffe import pytorch2caffe, plot_graph
from torchsummary import summary


def predict(smiles_list):
    model.eval()
    mol_prediction_list = []

    start =time.process_time()
    eval_loader = DataLoader(graph_dataset(smiles_list, graph_dict), 1, collate_fn=null_collate, num_workers=8, pin_memory=True, shuffle=False)	#
    end =time.process_time()
    print("whb main-predict")

    print ('1: Running dataloader time: %s Seconds'%(end-start))
    for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(eval_loader):
        shapes = [(16, 39), (34, 10), (34, 2), [16]]#(atom.shape, bond.shape, bond_index.shape, mol_index.shape)
        atom = atom.to(device)
        bond = bond.to(device)
        bond_index = bond_index.to(device)
        mol_index = mol_index.to(device)
        
        #print (atom.shape, bond.shape, bond_index.shape, mol_index.shape)
        #print (atom.dtype, bond.dtype, bond_index.dtype, mol_index.dtype)
        #print (model.cpu())
        #whb 
        summary(model, shapes, (atom, bond, bond_index, mol_index))
        start =time.process_time()


	# model start here
        mol_prediction = model(atom, bond, bond_index, mol_index)
        end =time.process_time()
        print ('2: Running model predict time: %s Seconds'%(end-start))


        mol_prediction_list.append(mol_prediction.data.squeeze().cpu().numpy())
    return mol_prediction_list


# In[7]:


test_predict_list = []
for fold in range(5):
    model = torch.load('saved_models/model_fold'+str(fold)+'_best.pt', map_location=torch.device('cpu'))
    test_predict_list.append(predict(smilesList))
    



start =time.process_time()

pred_from_five_model_test = np.array(test_predict_list).transpose()
stacking_model = pickle.load(open('saved_models/Ridge_regression_stacking_model.pt','rb'))
test_pred = stacking_model.predict(pred_from_five_model_test)    

end =time.process_time()
print ('3:Running stack_model predict time: %s Seconds'%(end-start))


std = 33.840
mean = 50.274
test_pred = test_pred*std+mean
print('The predicted bioavailability of ', args.smiles, test_pred)

