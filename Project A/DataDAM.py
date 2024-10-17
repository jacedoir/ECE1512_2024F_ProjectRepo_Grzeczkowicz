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
                 K=100, T=10, eta_S=0.1, zeta_S=1, eta_theta=0.01, zeta_theta=50, lambda_mmd=0.01, device='cuda', minibatches_size=64):
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
        self.minibatches_size = minibatches_size
        self.activations = {}



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
        sum = 0
        for i in range(0, final_feature_real.shape[0], self.IPC):
            sum += torch.norm(torch.mean(final_feature_real[i:i+self.IPC], dim=0) - torch.mean(final_feature_synthetic[i:i+self.IPC], dim=0))
        return sum
    
    
    def attention_loss(self, feature_real, feature_synthetic):
        """
        Compute the attention loss between real and synthetic data.
        """
        attention_real = get_attention(feature_real)
        attention_synthetic = get_attention(feature_synthetic)
        return torch.norm(attention_real - attention_synthetic)

    
    def getActivation(self,name):
        def hook_func(m, inp, op):
            self.activations[name] = op.clone()
        return hook_func
    
    ''' Defining the Refresh Function to store Activations and reset Collection '''
    def refreshActivations(self, activations):
        model_set_activations = [] # Jagged Tensor Creation
        for i in activations.keys():
            model_set_activations.append(activations[i])
        activations = {}
        return activations, model_set_activations
    
    ''' Defining the Delete Hook Function to collect Remove Hooks '''
    def delete_hooks(slef,hooks):
        for i in hooks:
            i.remove()
            return
    
    def attach_hooks(self,net):
        hooks = []
        base = net.module if torch.cuda.device_count() > 1 else net
        for module in (base.features.named_modules()):
            if isinstance(module[1], nn.ReLU):
                # Hook the Ouptus of a ReLU Layer
                hooks.append(base.features[int(module[0])].register_forward_hook(self.getActivation('ReLU_'+str(len(hooks)))))
        return hooks

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
                real_data = torch.zeros(self.minibatches_size, self.channels, self.im_size[0], self.im_size[1])
                syn_data = torch.zeros(self.minibatches_size, self.channels, self.im_size[0], self.im_size[1])
                for i in range(self.minibatches_size):
                    real_data[i] = self.images_all[np.random.choice(self.indices_class[c])]
                    if t == 0:
                        syn_data[i] = self.synthetic_dataset[c*self.IPC + np.random.choice(self.IPC)]
                    else:
                        syn_data[i] = self.saved_synthetic_dataset[t-1][0][c*self.IPC + np.random.choice(self.IPC)]
                real_data_labels = [c for i in range(self.minibatches_size)]
                syn_data_labels = [c for i in range(self.minibatches_size)]
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
            for k in tqdm(range(self.K), desc="Weight Initializations", leave=True):
                #get a random weight initialization
                net = get_network(self.model, self.channels, self.num_classes, self.im_size)
                net.to(self.device)
                optimizer = torch.optim.SGD(net.parameters(), lr=self.eta_theta)
                criterion = nn.CrossEntropyLoss()

                trainloader = torch.utils.data.DataLoader(TensorDataset(minibatches_real, minibatches_real_labels), batch_size=self.batch_size, shuffle=True)

                #train the network
                for step in range(self.zeta_theta):
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()
                        feature_real = net(inputs)
                        loss = criterion(feature_real, labels)
                        loss.backward()
                        optimizer.step()

                hooks = self.attach_hooks(net)

                feature_real_loader = torch.utils.data.DataLoader(TensorDataset(minibatches_real, minibatches_real_labels), batch_size=self.batch_size, shuffle=True)
                feature_synthetic_loader = torch.utils.data.DataLoader(TensorDataset(minibatches_synthetic, minibatches_synthetic_labels), batch_size=self.batch_size, shuffle=True)
                
                feature_real_list = []
                feature_synthetic_list = []
                for i, data in enumerate(feature_real_loader, 0):
                    inputs, labels = data
                    feature_real = net(inputs)
                    feature_real_list.append(feature_real)
                for i, data in enumerate(feature_synthetic_loader, 0):
                    inputs, labels = data
                    feature_synthetic = net(inputs)
                    feature_synthetic_list.append(feature_synthetic)
                feature_real = torch.cat(feature_real_list, dim=0)
                feature_real = feature_real[0].detach().cpu()
                feature_synthetic = torch.cat(feature_synthetic_list, dim=0)
                feature_synthetic = feature_synthetic[0]
                self.activations, original_model_set_activations = self.refreshActivations(self.activations)
                self.activations, syn_model_set_activations = self.refreshActivations(self.activations)
                self.delete_hooks(hooks)

                length_of_network = min(len(original_model_set_activations), len(syn_model_set_activations))
                for layer in range(length_of_network-1):
                
                    real_attention = get_attention(original_model_set_activations[layer].detach(), param=1, exp=1, norm='l2')
                    syn_attention = get_attention(syn_model_set_activations[layer], param=1, exp=1, norm='l2')
                    err = self.attention_loss(real_attention, syn_attention)
                    avg_attention_loss += err

                avg_mmd_loss += self.mmd_loss(feature_real, feature_synthetic)
                avg_attention_loss /= length_of_network-1

                del net
                torch.cuda.empty_cache()

            avg_attention_loss /= self.K
            avg_mmd_loss /= self.K

            #update the synthetic data for zeta_S steps
            for step in range(self.zeta_S):
                optimizer_images.zero_grad()
                loss = avg_attention_loss + self.lambda_mmd * avg_mmd_loss
                loss.backward()
                optimizer_images.step()

            minibatches_synthetic = minibatches_synthetic.detach().cpu()
            minibatches_synthetic_labels = minibatches_synthetic_labels.detach().cpu()
            minibatches_real = minibatches_real.detach().cpu()
            minibatches_real_labels = minibatches_real_labels.detach().cpu()

            #save the synthetic data
            self.saved_synthetic_dataset.append([self.synthetic_dataset.detach().cpu(), self.label_syn])

    def get_synthetic_dataset_step(self):
        return self.saved_synthetic_dataset
    
    def get_synthetic_dataset_final(self):
        return self.saved_synthetic_dataset[-1]



