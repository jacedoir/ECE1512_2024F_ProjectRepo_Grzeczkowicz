import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import *
from tqdm import tqdm
import torchvision




class PAD:
    def __init__(self, device, model, channel, num_classes, im_size, real_dataset, save_path, batch_size = 64, M=2,
                 K=100, T=10, eta_S=0.1, zeta_S=1, eta_theta=0.01, zeta_theta=50, alpha=0.01, minibatches_size=64):
        self.device = device
        self.model = model
        self.channel = channel
        self.num_classes = num_classes
        self.im_size = im_size
        self.real_dataset = real_dataset
        self.batch_size = batch_size
        self.K = K
        self.T = T
        self.eta_S = eta_S
        self.zeta_S = zeta_S
        self.eta_theta = eta_theta
        self.zeta_theta = zeta_theta
        self.alpha = alpha
        #self.minibatches_size = minibatches_size
        self.IPC = 10
        self.M = M
        self.save_path = save_path
        
        self.images_all = []
        self.labels_all = []
        self.indices_class = [[] for c in range(self.num_classes)]

        self.images_all = [torch.unsqueeze(self.real_dataset[i][0], dim=0) for i in range(len(self.real_dataset))]
        self.labels_all = [self.real_dataset[i][1] for i in range(len(self.real_dataset))]
        for i, label in enumerate(self.labels_all):
            self.indices_class[label].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(self.device)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=self.device)
        
        self.difficulty_scores = []
        self.saved_synthetic_dataset = []

    def difficulty_scoring(self):
        """
        Implementation of the Error L2-Norm scoring function
        """
        el2n_scores = []
        model = get_network(self.model, self.channel, self.num_classes, self.im_size)
        model.eval()  # Put model in evaluation mode
        with torch.no_grad():  # No need to compute gradients
            for i,inputs in enumerate(self.images_all):
                inputs, labels = inputs.to(self.device), self.labels_all[i].to(self.device)
                
                # Get model predictions
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
                
                # Calculate the L2-Norm (Euclidean distance) between predictions and true labels
                labels_onehot = F.one_hot(labels, num_classes=probabilities.shape[1]).float()
                el2n_score = torch.norm(probabilities - labels_onehot, p=2, dim=1).mean().item()
                
                #store score and id of the image
                el2n_scores.append([i,el2n_score])
        
        #sort the scores from least to greatest
        el2n_scores = np.array(el2n_scores)
        el2n_scores = el2n_scores[el2n_scores[:,1].argsort()]
        self.difficulty_scores = el2n_scores

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
    
    def schedule_data(self,t):
        nb_cuts = self.zeta_theta//2
        cut_size = self.images_all.shape[0]//nb_cuts
        if t <= nb_cuts:
            idx = self.difficulty_scores[:(t+1)*cut_size,0]
        else:
            idx = self.difficulty_scores[(t-nb_cuts)*cut_size:self.zeta_theta:,0]
        #convert to list
        idx = idx.tolist()
        return self.images_all[idx], self.labels_all[idx]
    
    def train(self, init="Gaussian", mean = 0, std = 1):
        # Initialize synthetic dataset
        if init == "real":
            synthetic_dataset, label_syn = self.initialize_synthetic_dataset_from_real()
        elif init == "Gaussian":
            synthetic_dataset, label_syn = self.initialize_synthetic_dataset_from_gaussian_noise(mean, std)
            
        synthetic_dataset.requires_grad = True
        
        stored_parameters = []
        
        model = get_network(self.model, self.channel, self.num_classes, self.im_size)
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer_net = torch.optim.SGD(model.parameters(), lr=self.eta_theta)
        model.train()
        stored_parameters.append(torch.clone(model.parameters()).detach())
        
        progress_train_net = tqdm(range(self.T), desc="Training the expert model")
        for t in progress_train_net:
            # Schedule data
            images, labels = self.schedule_data(t)
            # Train the expert model
            model.train()
            optimizer_net.zero_grad()
            outputs = model(images)[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_net.step()
            stored_parameters.append(torch.clone(model.parameters()).detach())
            progress_train_net.set_postfix(loss=loss.item())
            
        model.to('cpu')
        
        progress_train_data = tqdm(range(self.T), desc="Training the synthetic data")
        optimizer_syn = torch.optim.SGD([synthetic_dataset,], lr=self.eta_S)
        for t in progress_train_data:
            student_model = get_network(self.model, self.channel, self.num_classes, self.im_size)
            student_model.to(self.device)
            random_instant = np.random.randint(0, self.T-self.M)
            # train the student model random_instant + slef.T epochs
            student_model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(student_model.parameters(), lr=self.alpha)
            for i in range(0, random_instant+self.T):
                optimizer.zero_grad()
                outputs = student_model(synthetic_dataset)[1]
                loss = criterion(outputs, label_syn)
                loss.backward()
                optimizer.step()
            
            #get the parameters of the last layers
            num_layers = len(list(student_model.parameters()))
            k = torch.floor(torch.tensor(num_layers*self.alpha)).int()
            layer_kept = student_model.parameters()[k,:]
            # Update the synthetic data
            loss_syn = torch.norm(layer_kept - stored_parameters[random_instant+self.M], p=2)/torch.norm(stored_parameters[random_instant+self.M]-stored_parameters[random_instant], p=2)
            loss_syn.backward()
            optimizer_syn.step()
            
            # Save synthetic data
            self.saved_synthetic_dataset.append([torch.clone(synthetic_dataset).detach().cpu(), self.label_syn])

            # Save the last iteration to folder
            images, labels = self.saved_synthetic_dataset[-1]
            if not os.path.exists(self.save_path + "/step_" + str(t)):
                os.makedirs(self.save_path + "/step_" + str(t))
            for i in range(len(images)):
                torchvision.utils.save_image(images[i], self.save_path + "/step_" + str(t) + "/synthetic_" + str(i) + ".png")
        return synthetic_dataset

    def get_synthetic_dataset_step(self, step):
        return self.saved_synthetic_dataset[step]
    
    def get_synthetic_dataset_final(self):
        return self.saved_synthetic_dataset[-1]

