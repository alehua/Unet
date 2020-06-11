import os
import numpy as np
import time
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from network import U_net,Lite_U_net
from evaluation import *
from torchvision import transforms as T

class Solver(object):
    #def __init__(self,config,train_loader,valid_loader,test_loader):
    def __init__(self,train_loader):
        # Date loader
        self.train_loader = train_loader
        # self.valid_loader = valid_loader
        # self.test_loader = test_loader

        # Models
        self.img_ch    = 1                 # gray image
        self.output_ch = 1
        self.criterion = torch.nn.BCELoss()

        # Hyper-parameter
        self.lr       = 0.0002
        self.beta1    = 0.5        # momentum1 in Adam
        self.beta2    = 0.999      # momentum2 in Adam 

        # Training setting
        self.epochs        = 5
        self.epochs_decay  = 20
        self.batch_size    = 4

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()

    def build_model(self):
        #self.unet = U_net(self.img_ch,self.output_ch)
        self.unet = Lite_U_net(self.img_ch,self.output_ch)
        self.optimizer = optim.Adam(list(self.unet.parameters()),self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)
        # self.print_network(self.unet)

    def print_network(self, model):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()
    def presolve(self):
        """check the net work or not"""
        for i,(images,GT) in enumerate(self.train_loader):
            SR = self.unet(images)
            SR_probs = torch.sigmoid(GT)
            SR_flat = SR_probs.view(SR_probs.size(0),-1)
            GT_flat = GT.view(GT.size(0),-1)

            # loss = self.criterion(SR_flat,GT_flat)
            acc,prec,recall = get_param(SR,GT)
            print('Acc: %.4f, precision: %.4f, recall: %.4f' % (acc,prec,recall))

    def train(self):
        """Train encoder, generator and discriminator."""
        #====================================== Training ===========================================#
        #===========================================================================================#
        # No validate , NO test
        cnt = 1
        
        lr = self.lr
        best_score = 0

        for epoch in range(self.epochs):

            epoch_loss = 0
            acc = 0.	        # Accuracy
            precision = 0.		# precision 
            recall = 0.		    # Recall
            length = 0

            for i,(image,GT) in enumerate(self.train_loader):
                
                # GT : Ground Truth
                images = image.to(self.device)
                GT = GT.to(self.device)

                # SR : Segmentation Result
                SR = self.unet(images)
                SR_probs = torch.sigmoid(SR)
                SR_flat = SR_probs.view(SR_probs.size(0),-1)
                GT_flat = GT.view(GT.size(0),-1)

                loss = self.criterion(SR_flat,GT_flat)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                acc_,prec_,recall_ = get_param(SR,GT)
                acc       += acc_
                precision += prec_
                recall    += recall_
                length    += images.size(0)
            
            acc       = acc/length
            precision = precision/length
            recall    = recall/length
            
            # Print the log info
            print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, precision: %.4f, recall: %.4f' % (
                    epoch+1, self.epochs, \
                    epoch_loss,\
                    acc,precision,recall))
            unet_score = acc
            if unet_score > best_score or epoch == self.epochs-1:
                best_score = unet_score
                best_epoch = epoch
                best_unet = self.unet.state_dict()
                print('Best score : %.4f'%(best_score))

                unet_path = os.path.join("models//", '%d.pkl' %(cnt))
                cnt = cnt+1
                
                torch.save(best_unet,unet_path)
    def test(self,unet_path):
    #===================================== Test ====================================#
        # del self.unet
        self.unet.load_state_dict(torch.load(unet_path))
        self.unet.eval()
        dataiter = iter(self.train_loader)
        image,label = next(dataiter)
        images = image.to(self.device)
        SR = torch.sigmoid(self.unet(images))
        new_img = T.ToPILImage()(SR[0]).convert('RGB')
        new_img.show()
            

