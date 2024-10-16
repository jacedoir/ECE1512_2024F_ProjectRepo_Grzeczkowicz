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
    
    def _get_images(self, c, n): # get random n images from class c
        idx_shuffle = np.random.choice(self.indices_class[c], n, replace=False)
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

    
    def match_loss(self,gw_syn, gw_real, device, dis_metric):
        dis = torch.tensor(0.0, device=device)

        if dis_metric == 'ours':
            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += distance_wb(gwr, gws)

        elif dis_metric == 'mse':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

        elif dis_metric == 'cos':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        else:
            exit('unknown distance function: %s'%dis_metric)

        return dis
    
    def epoch(self,mode, dataloader, net, optimizer, criterion, device):
        loss_avg, acc_avg, num_exp = 0, 0, 0
        net = net.to(device)
        criterion = criterion.to(device)

        if mode == 'train':
            net.train()
        else:
            net.eval()

        for i_batch, datum in enumerate(dataloader):
            img = datum[0].float().to(device)
            lab = datum[1].long().to(device)
            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item()*n_b
            acc_avg += acc
            num_exp += n_b

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_avg /= num_exp
        acc_avg /= num_exp

        return loss_avg, acc_avg

    def run(self):
        # Optimizer for synthetic data samples
        optimizer_img = torch.optim.SGD([self.synthetic_dataset, ], lr=self.eta_S, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        total_steps = self.K * self.T * self.zeta_theta
        start_time = time.time()
        
        self.saved_synthetic_dataset.append([copy.deepcopy(self.synthetic_dataset.detach().cpu()), copy.deepcopy(self.label_syn.detach().cpu())])

        progress_bar_k = tqdm(range(self.K), desc="K", leave=True)
        for k in progress_bar_k:
            model = get_network(self.model, self.channels, self.num_classes, self.im_size).to(self.device)
            model.train()
            
            optimizer_model = torch.optim.SGD(model.parameters(), lr=self.eta_theta)
            optimizer_model.zero_grad()
            loss_avg, acc_avg = 0,0

            for t in range(self.T):
                loss = torch.tensor(0.0, device=self.device)
                
                for c in range(self.num_classes):
                    image_real = self._get_images(c, self.batch_size)
                    label_real = torch.ones((image_real.shape[0],), dtype=torch.long, device=self.device) * c
                    syntetic_image = self.synthetic_dataset[c*self.IPC:(c+1)*self.IPC].reshape((self.IPC, self.channels, self.im_size[0], self.im_size[1]))
                    syntetic_label = torch.ones((self.IPC,), dtype=torch.long, device=self.device) * c

                    output_real = model(image_real)
                    loss_real = criterion(output_real, label_real)
                    gw_real = torch.autograd.grad(loss_real, model.parameters())
                    gw_real = list((temp.detach().clone() for temp in gw_real))

                    output_syn = model(syntetic_image)
                    loss_syn = criterion(output_syn, syntetic_label)
                    gw_syn = torch.autograd.grad(loss_syn, list(model.parameters()), create_graph=True)
                    
                    loss += self.match_loss(gw_real, gw_syn, self.device, "ours")
                

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                synthetic_image_train, synthetic_label_train = copy.deepcopy(self.synthetic_dataset.detach().cpu()), copy.deepcopy(self.label_syn.detach().cpu())
                dst_synthetic_train = TensorDataset(synthetic_image_train, synthetic_label_train)
                train_loader = DataLoader(dst_synthetic_train, batch_size=self.batch_size, shuffle=True)

                for step in range(self.zeta_S):
                    loss_avg_model, acc_avg_model = self.epoch('train', train_loader, model, optimizer_model, criterion, self.device)

                loss_avg /= self.T*self.num_classes

                if (k+1) % 5 == 0:
                    #save every 5 epochs
                    self.saved_synthetic_dataset.append([copy.deepcopy(self.synthetic_dataset.detach().cpu()), copy.deepcopy(self.label_syn.detach().cpu())]) 

                if k == self.K-1:
                    #save the final synthetic dataset
                    self.saved_synthetic_dataset.append([copy.deepcopy(self.synthetic_dataset.detach().cpu()), copy.deepcopy(self.label_syn.detach().cpu())]) 
        return self.saved_synthetic_dataset

