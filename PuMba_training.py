#!/usr/bin/env python
# coding: utf-8

"""
# Reference: this code is based on PIsToN repository: https://github.com/stebliankin/piston
"""

import sys
import os
sys.path.append('/aul/homes/ashir018/PUMBA')
import numpy as np
import torch
# torch.cuda.set_device(1)
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.ViM_hybrid import Encoder
from networks.ViM_hybrid import ViM_Hybrid_encoder
from networks.ViM_hybrid import get_ml_config
from utils.trainer import fit_supCon, evaluate_val
from losses.proto_loss import ProtoLoss
from losses.supCon_loss import SupConLoss
from utils.utils import get_processed
from utils.dataset import PDB_complex_training

# Configuration
utils_path = './utils'
sys.path.append(utils_path)

DATA_DIR = os.getcwd() + '/data_preparation/'  # path to the preprocessed input data
TRAIN_LIST_FILE = './data/lists/training-list.txt'
VAL_LIST_FILE = './data/lists/val_list.txt'
MODEL_NAME = f'ViM_PuMba'
MODEL_DIR = f'./savedModels/{MODEL_NAME}'
IMG_SIZE = 32
MARGIN = 0
TEMP = 0.5
DIM = 16
PATIENCE = 2
SEED_ID = 7272
BATCH_SIZE = 1
MAX_EPOCH = 50
FEATURES_SUBSET = list(range(13))
N_FEATURES = len(FEATURES_SUBSET)


config = {}
config['dirs'] = {}
config['dirs']['data_prepare'] = DATA_DIR
print(DATA_DIR)
config['dirs']['grid'] = config['dirs']['data_prepare'] + '07-grid/'
print( "07-grid location is:", config['dirs']['grid'])
config['dirs']['docked'] = config['dirs']['data_prepare'] + 'docked/'
config['dirs']['tmp'] = '/aul/homes/ashir018/PUMBA/tmp'

config['ppi_const'] = {}
config['ppi_const']['patch_r'] = 16 # 16

os.environ["TMP"] = config['dirs']['tmp']
os.environ["TMPDIR"] = config['dirs']['tmp']
os.environ["TEMP"] = config['dirs']['tmp']


# Functions
def initialize_config():
    config = {}
    config['dirs'] = {
        'data_prepare': DATA_DIR,
        'grid': DATA_DIR + '07-grid/',
        'docked': DATA_DIR + 'docked/',
        'tmp': '/aul/homes/ashir018/PUMBA/tmp',
    }
    config['ppi_const'] = {'patch_r': 16}
    os.environ["TMP"] = config['dirs']['tmp']
    os.environ["TMPDIR"] = config['dirs']['tmp']
    os.environ["TEMP"] = config['dirs']['tmp']
    return config

def compute_mean_std(train_list, config):
    grid_native_list = []
    for ppi in train_list:
        # print(f"Loading grid for {ppi}...")
        grid_path = f"{config['dirs']['grid']}{ppi}.npy"
        if os.path.exists(grid_path):
            grid_native_list.append(np.load(grid_path, allow_pickle=True))
    print(f"Loaded {len(grid_native_list)} native complexes")
    all_grid = np.stack(grid_native_list, axis=0)

    radius = config['ppi_const']['patch_r']

    std_array = np.ones(N_FEATURES)
    mean_array = np.zeros(N_FEATURES)

    feature_pairs = {
        'shape_index': (0, 5),
        'ddc': (1, 6),
        'electrostatics': (2, 7),
        'charge': (3, 8),
        'hydrophobicity': (4, 9),
        'patch_dist': (10,),
        'SASA': (11, 12),
    }


    for feature in feature_pairs.keys():
        pixel_values = []
        for feature_i in feature_pairs[feature]:
            for image_i in tqdm(range(all_grid.shape[0])):
                for row_i in range(all_grid.shape[1]):
                    for column_i in range(all_grid.shape[2]):
                        x = column_i - radius
                        y = radius - row_i
                        if x**2 + y**2 < radius**2:
                            pixel_values.append(all_grid[image_i][row_i][column_i][feature_i])
        mean_value = np.mean(pixel_values)
        std_value = np.std(pixel_values)
        for feature_i in feature_pairs[feature]:
            mean_array[feature_i] = mean_value
            std_array[feature_i] = std_value

    print(f"Mean values: {mean_array}")
    print(f"Std values: {std_array}")
    return mean_array, std_array


def read_energies(energies_path, assign_zeros=False):
    
    """
    :param ppi:
    :return: numpy array of energy terms:
        (0) - indx
        (1) - Lrmsd     - ligand rmsd of the final position, after the rigid-body optimization.
        (2) -Irmsd     - interface rmsd of the final position, after the rigid-body optimization.
        (3) - st_Lrmsd  - initial ligand rmsd.
        (4) - st_Irmsd  - initial ligand rmsd.
    0 - (5) - glob      - global score of the candidate, which is linear combination of the terms described bellow. To rank the candidates, you should sort the rows by this column in ascending order.
    1 - (6) - aVdW      - attractive van der Waals
    2 - (7) - rVdW      - repulsive van der Waals
    3 - (8) - ACE       - Atomic Contact Energy
    4 - (9) - inside    - "Insideness" measure, which reflects the concavity of the interface.
    5 - (10) - aElec     - short-range attractive electrostatic term
    6 - (11) - rElec     - short-range repulsive electrostatic term
    7 - (12) - laElec    - long-range attractive electrostatic term
    8 - (13) - lrElec    - long-range repulsive electrostatic term
    9 - (14) - hb        - hydrogen and disulfide bonding
    10 - (15) - piS	  - pi-stacking interactions
    11 - (16) - catpiS	  - cation-pi interactions
    12 - (17) - aliph	  - aliphatic interactions
         (18) - prob      - rotamer probability

    """    
    to_read=False
    all_energies = None
    
    with open(energies_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            if to_read:
                if line.strip() == '':
                    print(f"Empty line found in {energies_path}. Assigning zeros.")
                    all_energies = np.zeros(13)
                    break

                if line.startswith('rob |'):
                    i += 1  # Skip to the next line after 'rob |'
                    if i < len(lines):
                        line = lines[i]
                    else:
                        # print(f"No line found after 'rob |' in {energies_path}. Assigning zeros.")
                        all_energies = np.zeros(13)
                        break

                all_energies = line.split('|')
                all_energies = [x.strip() for x in all_energies]
                all_energies = all_energies[5:18]
                all_energies = [float(x) for x in all_energies]
                all_energies = np.array(all_energies)
                break

            if 'Sol # |' in line:
                to_read = True
            i += 1
    if all_energies is not None:
        all_energies = np.nan_to_num(all_energies)
    elif assign_zeros:
        all_energies = np.zeros(13)       
    return all_energies


class PuMba_proto(nn.Module):

    def __init__(self, config, img_size=32, num_classes=2, zero_head=False, margin=0, temperature=0.1):
        super(PuMba_proto, self).__init__()

        """
        Input: Image with the following features:

        0 - Shape index | p1
        1 - Distance depended curvature | p1
        2 - Hydrogen bond potential | p1
        3 - Charge | p1
        4 - Hydrophobicity | p1
        5 - Shape index | p1
        6 - Distance depended curvature | p1
        7 - Hydrogen bond potential | p1
        8 - Charge | p1
        9 - Hydrophobicity | p1
        10 - Distance between atoms | p1 and p2
        11 - Relative ASA | p1
        12 - Relative ASA | p1

        Energies with the following features:
        0  - glob      - global score of the candidate, which is linear combination of the terms described bellow. To rank the candidates, you should sort the rows by this column in ascending order.
        1  - aVdW      - attractive van der Waals
        2  - rVdW      - repulsive van der Waals
        3  - ACE       - Atomic Contact Energy | desolvation (10.1006/jmbi.1996.0859)
        4  - inside    - "Insideness" measure, which reflects the concavity of the interface.
        5  - aElec     - short-range attractive electrostatic term
        6  - rElec     - short-range repulsive electrostatic term
        7  - laElec    - long-range attractive electrostatic term
        8  - lrElec    - long-range repulsive electrostatic term
        9  - hb        - hydrogen and disulfide bonding
        10 - piS	     - pi-stacking interactions
        11 - catpiS	  - cation-pi interactions
        12 - aliph	  - aliphatic interactions
        Loss = alpha * (BCE_i/5) + Global_BCE, where alpha is a hyperparameter from 0 to 1
        """
        self.index_dict = {
            'shape_complementarity': ( (0, 5, 1, 6, 10), (1,2,4) ),
            'RASA': ( (11, 12, 10), (3,) ),
            'hydrogen_bonds': ( (2, 7, 10), (9,) ),
            'charge': ( (3, 8, 10), (5, 6, 7, 8, 10, 11, 12) ),
            'hydrophobicity': ((4, 9, 10), ())
        }

        self.img_size = img_size
        self.num_classes = num_classes
        self.zero_head = zero_head


        self.spatial_transformers_list = nn.ModuleList()

        for feature in self.index_dict.keys():
            self.spatial_transformers_list.append(self.init_vim(config, channels=len(self.index_dict[feature][0]),
                                                                n_individual=len(self.index_dict[feature][1])))

        self.classifier = config.classifier  
        self.feature_transformer = Encoder(config)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size), requires_grad=True)
        self.proto_vector_pos = nn.Parameter(torch.rand(1,config.hidden_size), requires_grad=True)
        self.proto_vector_neg = nn.Parameter(torch.rand(1,config.hidden_size), requires_grad=True)
        self.margin = margin
        self.temperature = temperature


    def init_vim(self, config, channels, n_individual):
        """
        Initialize ViM Network for a given tupe of features
        :param model_config:
        :param channels:
        :param n_individual:
        :return:
        """
        return ViM_Hybrid_encoder(config, n_individual, img_size=self.img_size,
                        num_classes=self.num_classes, channels=channels, vis=True)

    def forward(self, img, energies, labels=None):

        all_x = []
        all_spatial_attn = []
        for i, feature in enumerate(self.index_dict.keys()):
            img_tmp = img[:,self.index_dict[feature][0],:,:]
            energy_tmp = energies[:, self.index_dict[feature][1]]
            x, attn = self.spatial_transformers_list[i](img_tmp, energy_tmp)
            all_x.append(x)
            all_spatial_attn.append(attn)

        x = torch.stack(all_x, dim=1) 
        B = x.shape[0] #batch
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.feature_transformer(x)  
        x = x[:, 0] 
        x = nn.functional.normalize(x) 

        proto_vector_pos = nn.functional.normalize(self.proto_vector_pos)
        proto_vector_neg = nn.functional.normalize(self.proto_vector_neg)
        dist = nn.PairwiseDistance()
        dist_to_pos_prototype = dist(x, proto_vector_pos.repeat(x.shape[0], 1))
        dist_to_neg_prototype = dist(x, proto_vector_neg.repeat(x.shape[0], 1))

        logits = torch.stack([dist_to_pos_prototype-dist_to_neg_prototype, dist_to_neg_prototype-dist_to_pos_prototype], axis=1)

        scores = dist_to_pos_prototype-dist_to_neg_prototype
        if labels is not None:
            proto_loss_fn = ProtoLoss(margin=self.margin)
            bce_loss_fn = nn.CrossEntropyLoss()
            SupConLoss_fn = SupConLoss(temperature=self.temperature, base_temperature=self.temperature*10)

            proto_loss = proto_loss_fn(x, proto_vector_pos, proto_vector_neg, labels)
            BCE_loss = bce_loss_fn(logits, labels) 
            supCon_loss = SupConLoss_fn(x, labels)
            loss = proto_loss + BCE_loss + supCon_loss

            return scores, all_spatial_attn, loss
        else:
            return scores, all_spatial_attn
        
        
def train_pumba(search_space, train_list, val_list, SEED_ID, IMG_SIZE, PATIENCE, MODEL_DIR,
                 MODEL_NAME, docked_dir, pos_grid_dir, std, mean, energies_std, energies_mean,
                 MAX_EPOCHS=50, N_FEATURES=10, feature_subset=None, disable_tqdm=True, print_summary=False, data_prepare_dir='/data_preparation/'):
    train_db = PDB_complex_training(
        train_list, training_mode=True, feature_subset=feature_subset,
        data_prepare_dir=data_prepare_dir, neg_pos_ratio=search_space['neg_pos_ratio'],
        mean=mean, std=std, energies_mean=energies_mean, energies_std=energies_std)

    val_db = PDB_complex_training(
        val_list, training_mode=False, feature_subset=feature_subset,
        data_prepare_dir=data_prepare_dir, neg_pos_ratio=search_space['neg_pos_ratio'],
        mean=mean, std=std, energies_mean=energies_mean, energies_std=energies_std)

    trainloader = DataLoader(train_db, batch_size=1, shuffle=True, pin_memory=True)
    valloader = DataLoader(val_db, batch_size=1, shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = get_ml_config(search_space)
    model = PuMba_proto(
        model_config, img_size=IMG_SIZE, temperature=search_space['temperature'],
        num_classes=2, margin=search_space['margin']).float()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=search_space['lr'], weight_decay=search_space['weight_decay'])
    model, history, saved_index = fit_supCon(
        MAX_EPOCHS, model, trainloader, valloader, optimizer, model_name=MODEL_NAME,
        image_size=IMG_SIZE, channels=N_FEATURES, device=device, save_model=True,
        saved_model_dir=MODEL_DIR, patience=PATIENCE, print_summary=print_summary,
        disable_tqdm=disable_tqdm, include_energy=True, include_attn=True, inside_loss=True,
        n_individual=len(energies_mean))
    val_loss, val_auc = evaluate_val(valloader, model, device, include_energy=True, include_attn=True, inside_loss=True)
    return model, history, saved_index


def main():
    config = initialize_config()
    print("Configuration initialized:", config)
    train_list = [x.strip('\n') for x in open(TRAIN_LIST_FILE, 'r').readlines()]
    val_list = [x.strip('\n') for x in open(VAL_LIST_FILE, 'r').readlines()]
    train_list_updated = get_processed(train_list, config)
    val_list_updated = get_processed(val_list, config)
    print(f"Training on {len(train_list_updated)} complexes, validation on {len(val_list_updated)} complexes")
    
    mean_array, std_array = compute_mean_std(train_list_updated, config)

    all_energies_list = []
    for ppi in train_list_updated:
        energy_path = f"{config['dirs']['grid']}/refined-out-{ppi}.ref"
        # print(energy_path)
        if not os.path.exists(energy_path):
            print(f"Energy file {energy_path} does not exist, skipping...")
        if os.path.exists(energy_path):
            energy_i = read_energies(energy_path)
            if energy_i is not None:
                all_energies_list.append(energy_i)
            if energy_i is None:
                print(f"Energy file {energy_path} is empty, assigning zeros...")

    print(f"Loaded energy terms from {len(all_energies_list)} native complexes")

    all_energies = np.stack(all_energies_list, axis=0)
    all_energies_mean = np.mean(all_energies, axis=0)
    all_energies_std = np.std(all_energies, axis=0)
    print(f"Energies mean: {list(all_energies_mean)}")
    print(f"Energies std: {list(all_energies_std)}")

    params = {'dim_head': DIM, 'hidden_size': DIM, 'dropout': 0, 'attn_dropout': 0, 'lr': 0.0001,
              'n_heads': 8, 'neg_pos_ratio': 5, 'patch_size': 4, 'transformer_depth': 8,
              'weight_decay': 0.0001, 'margin': MARGIN, 'temperature': TEMP}

    os.makedirs(MODEL_DIR, exist_ok=True)

    model, history, saved_index = train_pumba(
        params, train_list=train_list_updated, val_list=val_list_updated, SEED_ID=SEED_ID,
        IMG_SIZE=IMG_SIZE, PATIENCE=PATIENCE, MODEL_NAME=MODEL_NAME, MAX_EPOCHS=MAX_EPOCH,
        N_FEATURES=N_FEATURES, MODEL_DIR=MODEL_DIR, docked_dir=config['dirs']['docked'],
        pos_grid_dir=config['dirs']['grid'], mean=mean_array, std=std_array,
        energies_mean=all_energies_mean, energies_std=all_energies_std, feature_subset=FEATURES_SUBSET)

if __name__ == "__main__":
    main()
