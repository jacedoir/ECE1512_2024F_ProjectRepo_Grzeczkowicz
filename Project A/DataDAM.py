from tqdm import tqdm
from networks import ConvNet
from utils import *
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset

class DataDAM:
    def __init__(self, model, real_dataset, IPC, num_classes, im_size, channels, save_path, batch_size = 64,
                 K=100, T=10, eta_S=0.1, zeta_S=1, eta_theta=0.01, zeta_theta=50, lambda_mmd=0.01, device='cuda'):
        """ 
        This class aims to reproduce the DataDAM algorithm for synthesizing a dataset. It is an adaptation of the original code from https://github.com/DataDistillation/DataDAM/blob/main/main_DataDAM.py#L75 Some changes have been made to adapt to the current framework.
        """
        self.model = model
        self.real_dataset = real_dataset
        self.IPC = IPC
        self.num_classes = num_classes
        self.im_size = im_size
        self.channels = channels
        self.K = K  # Number of random weight initializations
        self.T = T  # Number of iterations
        self.eta_S = eta_S  # Learning rate for synthetic samples
        self.zeta_S = zeta_S  # Steps for optimizing synthetic samples
        self.eta_theta = eta_theta  # Learning rate for the model
        self.zeta_theta = zeta_theta  # Steps for optimizing the model
        self.lambda_mmd = lambda_mmd  # Task balance parameter
        self.device = device
        self.saved_synthetic_dataset = []
        self.save_path = save_path
        self.batch_size = batch_size



        self.images_all = []
        self.labels_all = []
        self.indices_class = [[] for c in range(self.num_classes)]

        self.images_all = [torch.unsqueeze(self.real_dataset[i][0], dim=0) for i in range(len(self.real_dataset))]
        self.labels_all = [self.real_dataset[i][1] for i in range(len(self.real_dataset))]
        for i, label in enumerate(self.labels_all):
            self.indices_class[label].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(self.device)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=self.device)
    
    def _get_images(self, n): # get random n images from class c
        idx_shuffle = np.random.choice(self.indices_class, n, replace=False)
        return self.images_all[idx_shuffle]

    def initialize_synthetic_dataset_from_real(self):
        #Sample IPC images per class
        self.synthetic_dataset = []
        for c in range(self.num_classes):
            self.synthetic_dataset.append(self._get_images(c, self.IPC).detach().data)
        self.synthetic_dataset = torch.cat(self.synthetic_dataset, dim=0)
        self.label_syn = torch.tensor([np.ones(self.IPC)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    
    def initialize_synthetic_dataset_from_gaussian_noise(self, mean, std):
        #Create IPC images per class from Gaussian noise
        self.synthetic_dataset = torch.normal(mean, std, size=(self.num_classes*self.IPC, self.channels, self.im_size[0], self.im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
        self.label_syn = torch.tensor([np.ones(self.IPC)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    def mmd_loss(self, final_feature_real, final_feature_synthetic):
        """
        Compute the MMD loss between real and synthetic data.
        """
        print(final_feature_real.shape)
        print(final_feature_synthetic.shape)
        return
    
    def attention_loss(self, feature_real, feature_synthetic):
        """
        Compute the attention loss between real and synthetic data.
        """
        return

    def train(self):
        """
        Train the model.
        """
        temp_net = get_network(self.model, self.channels, self.num_classes, self.im_size)
        temp_net.to(self.device)
        optimizer_images = torch.optim.SGD(temp_net.parameters(), lr=self.eta_S)
        for t in tqdm(range(self.T), desc="Iterations", leave=True):
            #number of iterations
            #sample minibatches for real and synthetic data for each class c
            minibatches_real = []
            minibatches_synthetic = []
            minibatches_real_labels = []
            minibatches_synthetic_labels = []
            for c in range(self.num_classes):
                real_data = self.images_all[np.random.choice(self.indices_class[c], self.batch_size, replace=True)]
                syn_data = self.synthetic_dataset[np.random.choice(self.indices_class[c], self.batch_size, replace=True)]
                real_data_labels = [c for i in range(self.batch_size)]
                syn_data_labels = [c for i in range(self.batch_size)]
                minibatches_real.append(real_data)
                minibatches_synthetic.append(syn_data)
                minibatches_real_labels.append(real_data_labels)
                minibatches_synthetic_labels.append(syn_data_labels)
            minibatches_real = torch.cat(minibatches_real, dim=0).to(self.device)
            minibatches_synthetic = torch.cat(minibatches_synthetic, dim=0).to(self.device)
            minibatches_real_labels = torch.tensor(minibatches_real_labels, dtype=torch.long, requires_grad=False, device=self.device).view(-1)
            minibatches_synthetic_labels = torch.tensor(minibatches_synthetic_labels, dtype=torch.long, requires_grad=False, device=self.device).view(-1)


            avg_attention_loss = 0
            avg_mmd_loss = 0
            for k in range(self.K):
                #get a random weight initialization
                net = get_network(self.model, self.channels, self.num_classes, self.im_size)
                net.to(self.device)
                optimizer = torch.optim.SGD(net.parameters(), lr=self.eta_theta)
                criterion = nn.MSELoss()

                #train the network
                for step in range(self.zeta_theta):
                    optimizer.zero_grad()
                    feature_real = net(minibatches_real)
                    loss = criterion(feature_real, minibatches_synthetic_labels)
                    loss.backward()
                    optimizer.step()

                feature_real = net(minibatches_real)
                feature_synthetic = net(minibatches_synthetic)
                avg_attention_loss += self.attention_loss(feature_real, feature_synthetic)
                avg_mmd_loss += self.mmd_loss(feature_real[-1], feature_synthetic[-1])

            avg_attention_loss /= self.K
            avg_mmd_loss /= self.K

            #update the synthetic data for zeta_S steps
            for step in range(self.zeta_S):
                optimizer_images.zero_grad()
                loss = avg_attention_loss + self.lambda_mmd * avg_mmd_loss
                loss.backward()
                optimizer_images.step()

            #save the synthetic data
            self.saved_synthetic_dataset.append([self.synthetic_dataset.detach().cpu(), self.label_syn])

    def get_synthetic_dataset_step(self):
        return self.saved_synthetic_dataset
    
    def get_synthetic_dataset_final(self):
        return self.saved_synthetic_dataset[-1]



