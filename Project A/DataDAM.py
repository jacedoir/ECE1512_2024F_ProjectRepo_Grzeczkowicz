import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from networks import ConvNet
from tqdm import tqdm
import copy
import numpy as np
class DataDAM:
    def __init__(self, model, real_dataset, IPC, num_classes, im_size, channels, init="Sample", 
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
        self.init = init
        self.K = K  # Number of random weight initializations
        self.T = T  # Number of iterations
        self.eta_S = eta_S  # Learning rate for synthetic samples
        self.zeta_S = zeta_S  # Steps for optimizing synthetic samples
        self.eta_theta = eta_theta  # Learning rate for the model
        self.zeta_theta = zeta_theta  # Steps for optimizing the model
        self.lambda_mmd = lambda_mmd  # Task balance parameter
        self.device = device
        
        self.images_all = []
        self.labels_all = []
        self.indices_class = [[] for c in range(num_classes)]
        
        self.images_all = [torch.unsqueeze(self.real_dataset[i][0], dim=0) for i in range(len(self.real_dataset))]
        self.labels_all = [self.real_dataset[i][1] for i in range(len(self.real_dataset))]
        for i, lab in enumerate(self.labels_all):
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(self.device)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=self.device)

        self.eval_it_pool = [self.T] # The list of iterations when we evaluate models and record results.
        self.model_eval_pool = [self.model]



        
    
    def initialize_synthetic_dataset(self):
        """Initialize the synthetic dataset with either noise or sampled real images."""
        self.image_syn = torch.randn(size=(self.num_classes*self.IPC, self.channels, self.im_size[0], self.im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
        self.label_syn = torch.tensor([np.ones(self.IPC)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        
        if self.init == "Sample":
            print('initialize synthetic data from random real images')
            for c in range(self.num_classes):
                self.image_syn.data[c*self.IPC:(c+1)*self.IPC] = self._get_images(c, self.IPC).detach().data
        elif self.init == "Noise":
            print('initialize synthetic data from random noise')

        # Now that synthetic dataset exists, set up its optimizer
        self.optimizer_S = torch.optim.SGD([{'params': self.synthetic_dataset}], lr=self.eta_S)
    
    def _get_images(self, c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_shuffle]

    def train(self):
        """Main training loop to learn the synthetic dataset."""
        self.optimizer_img = torch.optim.SGD([self.image_syn, ], lr=self.eta_S, momentum=0.5) # optimizer_img for synthetic data
        self.optimizer_img.zero_grad()
        print('training begins')
        
        ''' Defining the Hook Function to collect Activations '''
        activations = {}
        def getActivation(name):
            def hook_func(m, inp, op):
                activations[name] = op.clone()
            return hook_func
        
        ''' Defining the Refresh Function to store Activations and reset Collection '''
        def refreshActivations(activations):
            model_set_activations = [] # Jagged Tensor Creation
            for i in activations.keys():
                model_set_activations.append(activations[i])
            activations = {}
            return activations, model_set_activations
        
        ''' Defining the Delete Hook Function to collect Remove Hooks '''
        def delete_hooks(hooks):
            for i in hooks:
                i.remove()
            return
        
        def attach_hooks(net):
            hooks = []
            base = net.module if torch.cuda.device_count() > 1 else net
            for module in (base.features.named_modules()):
                if isinstance(module[1], nn.ReLU):
                    # Hook the Ouptus of a ReLU Layer
                    hooks.append(base.features[int(module[0])].register_forward_hook(getActivation('ReLU_'+str(len(hooks)))))
            return hooks
        

        max_mean = 0
        for it in range(self.T+1):

            ''' Evaluate synthetic data '''
            if it in self.eval_it_pool:
                for model_eval in self.model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    Start = time.time()
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, self.channels, self.num_classes, self.im_size).to(self.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(self.image_syn.detach()), copy.deepcopy(self.label_syn.detach()) # avoid any unaware modification
                        mini_net, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        if acc_test > best_5[-1]:
                            best_5[-1] = acc_test
                    
                    Finish = (time.time() - Start)/10
                    
                    print("TOTAL TIME WAS: ", Finish)
                            
                            
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    if np.mean(accs) > max_mean:
                        data=[]
                        data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                        torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc_.pt'%(args.method, args.dataset, args.model, args.ipc)))
                    # Track All of them!
                    total_mean[exp]['mean'].append(np.mean(accs))
                    total_mean[exp]['std'].append(np.std(accs))
                    
                    accuracy_logging["mean"].append(np.mean(accs))
                    accuracy_logging["std"].append(np.std(accs))
                    accuracy_logging["max_mean"].append(np.max(accs))
                    
                    
                    if it == self.T: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                # save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                # image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                # for ch in range(channel):
                #     image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                # image_syn_vis[image_syn_vis<0] = 0.0
                # image_syn_vis[image_syn_vis>1] = 1.0
                # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False
                    
            loss_avg = 0
            def error(real, syn, err_type="MSE"):
                        
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
                    err = torch.sum((torch.mean(real.reshape(num_classes, args.batch_real, -1), dim=1).cpu() - torch.mean(syn.cpu().reshape(num_classes, args.ipc, -1), dim=1))**2)
                elif(err_type == "MAE_B"):
                    err = torch.sum(torch.abs(torch.mean(real.reshape(num_classes, args.batch_real, -1), dim=1).cpu() - torch.mean(syn.reshape(num_classes, args.ipc, -1).cpu(), dim=1)))
                elif (err_type == "ANG_B"):
                    rl = torch.mean(real.reshape(num_classes, args.batch_real, -1), dim=1).cpu()
                    sy = torch.mean(syn.reshape(num_classes, args.ipc, -1), dim=1)
                    
                    denom = (torch.sum(rl**2)**0.5).cpu() * (torch.sum(sy**2)**0.5).cpu()
                    num = rl.cpu() * sy.cpu()
                    err = torch.sum(torch.acos(num/denom))
                return err
            
            ''' update synthetic data '''
            loss = torch.tensor(0.0)
            mid_loss = 0
            out_loss = 0

            images_real_all = []
            images_syn_all = []
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                images_real_all.append(img_real)
                images_syn_all.append(img_syn)

            images_real_all = torch.cat(images_real_all, dim=0)
            
            images_syn_all = torch.cat(images_syn_all, dim=0)

            
            hooks = attach_hooks(net)
            
            output_real = net(images_real_all)[0].detach()
            activations, original_model_set_activations = refreshActivations(activations)
            
            output_syn = net(images_syn_all)[0]
            activations, syn_model_set_activations = refreshActivations(activations)
            delete_hooks(hooks)
            
            length_of_network = len(original_model_set_activations)# of Feature Map Sets
            
            for layer in range(length_of_network-1):
                
                real_attention = get_attention(original_model_set_activations[layer].detach(), param=1, exp=1, norm='l2')
                syn_attention = get_attention(syn_model_set_activations[layer], param=1, exp=1, norm='l2')

                tl =  100*error(real_attention, syn_attention, err_type="MSE_B")
                loss+=tl
                mid_loss += tl

            output_loss =  100*args.task_balance * error(output_real, output_syn, err_type="MSE_B")
            
            loss += output_loss
            out_loss += output_loss

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            torch.cuda.empty_cache()

            loss_avg /= (num_classes)
            out_loss /= (num_classes)
            mid_loss /= (num_classes)
            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))


    def _random_weight_initialization(self, layer):
        """Random weight initialization for layers."""
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def _get_real_batch(self):
        """Fetch a batch of real data."""
        real_indices = torch.randperm(len(self.real_dataset))[:self.IPC]
        real_images = torch.stack([self.real_dataset[idx][0] for idx in real_indices]).to(self.device)
        real_labels = torch.tensor([self.real_dataset[idx][1] for idx in real_indices], dtype=torch.long).to(self.device)
        return real_images, real_labels

    def _get_synthetic_batch(self):
        """Fetch a batch of synthetic data."""
        return next(iter(self.synthetic_dataset))

    def _forward_pass(self, model, real_batch, synth_batch):
        """Perform forward pass on both real and synthetic batches."""
        real_images, real_labels = real_batch
        synth_images, synth_labels = synth_batch

        # Forward pass
        real_features = model(real_images)
        synth_features = model(synth_images)

        # Compute attention maps
        real_attention = self._compute_attention_maps(real_features)
        synth_attention = self._compute_attention_maps(synth_features)

        # Compute losses
        real_loss = self._compute_mmd_loss(real_features[-1], synth_features[-1])
        synth_loss = sum([torch.nn.functional.mse_loss(r, s) for r, s in zip(real_attention, synth_attention)])
        
        return real_loss, synth_loss

    def _compute_attention_maps(self, features):
        """Generate attention maps by summing over channel dimensions."""
        return [torch.sum(torch.abs(fmap), dim=1) for fmap in features]

    def _compute_mmd_loss(self, real_features, synth_features):
        """Compute MMD loss between real and synthetic features."""
        mean_real = real_features.mean(dim=0)
        mean_synth = synth_features.mean(dim=0)
        return torch.nn.functional.mse_loss(mean_real, mean_synth)
    
    def get_condensed_dataset(self):
        """Return the condensed (synthetic) dataset after training."""
        return self.synthetic_dataset
