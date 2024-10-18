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
import torchvision
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
    
    def get_images(self, c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_shuffle]

    def initialize_synthetic_dataset_from_real(self):
        #Sample IPC images per class
        synthetic_dataset = torch.randn(size=(self.num_classes*self.IPC, self.channels, self.im_size[0], self.im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
        for c in range(self.num_classes):
            synthetic_dataset.data[c*self.IPC:(c+1)*self.IPC] = self.get_images(c, self.IPC).detach().data
        label_syn = torch.tensor([np.ones(self.IPC)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        return synthetic_dataset, label_syn
    def initialize_synthetic_dataset_from_gaussian_noise(self, mean, std):
        #Create IPC images per class from Gaussian noise
        synthetic_dataset = torch.normal(mean, std, size=(self.num_classes*self.IPC, self.channels, self.im_size[0], self.im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
        label_syn = torch.tensor([np.ones(self.IPC)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        return synthetic_dataset, label_syn
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
    
    def error(self,real, syn, err_type="MSE"):       
        if(err_type == "MSE"):
            err = torch.sum((torch.mean(real, dim=0) - torch.mean(syn, dim=0))**2)
        
        elif (err_type == "MAE"):
            err = torch.sum(torch.abs(torch.mean(real, dim=0) - torch.mean(syn, dim=0)))
            
        elif (err_type == "ANG"):
            rl = torch.mean(real, dim=0) 
            sy = torch.mean(syn, dim=0)
            num = torch.matmul(rl, sy)
            denom = (torch.sum(rl**2)**0.5) * (torch.sum(sy**2)**0.5)
            err = torch.acos(num/denom)
            
        elif(err_type == "MSE_B"):
            err = torch.sum((torch.mean(real.reshape(self.num_classes, self.minibatches_size,-1), dim=1).cpu() - torch.mean(syn.cpu().reshape(self.num_classes, self.IPC, -1), dim=1))**2)
        return err

    def train(self, init="Gaussian", mean = 0, std = 1):
        """
        Train the model, adapted to the style of the first method.
        """
        if init=="Gaussian":
            syn_dataset, label_syn = self.initialize_synthetic_dataset_from_gaussian_noise(mean, std)
        elif init=="Real":
            syn_dataset, label_syn = self.initialize_synthetic_dataset_from_real()
        else:
            raise ValueError("Invalid initialization type. Please specify 'Gaussian' or 'Real'.")

        self.synthetic_dataset = copy.deepcopy(syn_dataset).detach().cpu()
        self.label_syn = copy.deepcopy(label_syn).detach().cpu()

        # Save the orginal synthetic data to folder
        if not os.path.exists(self.save_path + "/step_0"):
            os.makedirs(self.save_path + "/step_0")
        for i in range(len(self.synthetic_dataset)):
            torchvision.utils.save_image(self.synthetic_dataset[i], self.save_path + "/step_0" + "/synthetic_" + str(i) + ".png")

        syn_dataset = syn_dataset.to(self.device).requires_grad_(True)


        torch.autograd.set_detect_anomaly(True)
        # image_syn_all = []
        # for c in range(self.num_classes):
        #     image_syn_all.append(self.synthetic_dataset[c*self.IPC:(c+1)*self.IPC].reshape(self.IPC, self.channels, self.im_size[0], self.im_size[1]))
        # image_syn_all = torch.cat(image_syn_all, dim=0).to(self.device).requires_grad_(True)

        optimizer_images = torch.optim.SGD([syn_dataset], lr=self.eta_S)

        for t in range(1,self.T+1):
            loss = torch.tensor(0.0)
            mid_loss = 0
            out_loss = 0

            optimizer_images.zero_grad()

            # image_syn_all = []
            # for c in range(self.num_classes):
            #     image_syn_all.append(self.synthetic_dataset[c*self.IPC:(c+1)*self.IPC].reshape(self.IPC, self.channels, self.im_size[0], self.im_size[1]))
            # image_syn_all = torch.cat(image_syn_all, dim=0)

            # Sample minibatches for real data
            minibatches_real = []
            minibatches_real_labels = []
            for c in range(self.num_classes):
                real_data = torch.zeros(self.minibatches_size, self.channels, self.im_size[0], self.im_size[1])
                for i in range(self.minibatches_size):
                    real_data[i] = self.images_all[np.random.choice(self.indices_class[c])]
                real_data_labels = [c for i in range(self.minibatches_size)]
                minibatches_real.append(real_data)
                minibatches_real_labels.append(real_data_labels)
            
            minibatches_real = torch.cat(minibatches_real, dim=0).to(self.device).requires_grad_(False)
            minibatches_real_labels = torch.tensor(minibatches_real_labels, dtype=torch.long, device=self.device).view(-1)

            progress_k = tqdm(range(self.K), desc=f"Iteration {t}/{self.T} - Weight Initializations")

            # Reinitialize network weights for each iteration
            # net = get_network(self.model, self.channels, self.num_classes, self.im_size)
            # net.to(self.device)
            

            # optimizer_net = optim.SGD(net.parameters(), lr=self.eta_theta)
            # criterion = nn.CrossEntropyLoss()
    

            for k in progress_k:
                net = get_network(self.model, self.channels, self.num_classes, self.im_size)
                net.to(self.device) 
                net.train()

                # # Training the network over zeta_theta iterations (like the first method)
                # for step in range(self.zeta_theta):
                #     optimizer_net.zero_grad()
                #     output_real = net(minibatches_real)
                #     loss_real = criterion(output_real, minibatches_real_labels)
                #     loss_real.backward()
                #     optimizer_net.step()

                # net.eval()  # Evaluation phase

                # Attach hooks to get intermediate activations
                hooks = self.attach_hooks(net)

                # Forward pass for real data
                output_real = net(minibatches_real)
                self.activations, original_model_set_activations = self.refreshActivations(self.activations)
            

                # Forward pass for synthetic data
                output_syn = net(syn_dataset)
                self.activations, syn_model_set_activations = self.refreshActivations(self.activations)

                # Detach outputs to avoid computing gradients on them further
                output_real = output_real[0].detach()
                output_syn = output_syn[0]

                self.delete_hooks(hooks)

                # Calculate layer-wise attention loss, similar to the first method
                length_of_network = len(original_model_set_activations)
                
                for layer in range(length_of_network - 1):
                    real_attention = get_attention(original_model_set_activations[layer].detach(), param=1, exp=1, norm='l2')
                    syn_attention = get_attention(syn_model_set_activations[layer], param=1, exp=1, norm='l2')
                    tl = 10000*self.error(real_attention, syn_attention, err_type="MSE_B")
                    loss += tl
                    mid_loss += tl

                # Calculate output loss using MMD (as in the first method)
                output_loss = 10000*self.lambda_mmd * self.error(output_real, output_syn, err_type="MSE_B")
                loss += output_loss
                out_loss += output_loss
            
            loss /= self.K
            out_loss /= self.K
            mid_loss /= self.K
            
            loss.backward()
            optimizer_images.step()
            print('Loss: ', loss.item(), "out_loss: ", out_loss.item(), "mid_loss: ", mid_loss.item())
            torch.cuda.empty_cache()



            # Save synthetic data
            self.saved_synthetic_dataset.append([torch.clone(syn_dataset).detach().cpu(), self.label_syn])

            # Save the last iteration to folder
            images, labels = self.saved_synthetic_dataset[-1]
            if not os.path.exists(self.save_path + "/step_" + str(t)):
                os.makedirs(self.save_path + "/step_" + str(t))
            for i in range(len(images)):
                torchvision.utils.save_image(images[i], self.save_path + "/step_" + str(t) + "/synthetic_" + str(i) + ".png")

            

    def get_synthetic_dataset_step(self, step):
        return self.saved_synthetic_dataset[step]
    
    def get_synthetic_dataset_final(self):
        return self.saved_synthetic_dataset[-1]



