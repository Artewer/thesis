
# Libs
import numpy as np
import logging
import matplotlib.pyplot as plt
# from tensorflow.keras import models,layers,regularizers,optimizers
import torch.nn as nn
import torch
import torch.optim as optim



# %% NN Class for each bidder


class BidderNN(nn.Module):
    def __init__(self, M, model_parameters):
        super(BidderNN, self).__init__()

        self.model_parameters = model_parameters
        self.M = M
        architecture = self.model_parameters['architecture']
        dropout = self.model_parameters['dropout']
        dp = self.model_parameters['dropout_prob']

        architecture = [int(layer) for layer in architecture]  # integer check
        number_of_hidden_layers = len(architecture)
        dropout = bool(dropout)
        # -------------------------------------------------- NN Architecture -------------------------------------------------#
        # GET MODEL HERE
        # first hidden layer
        model = nn.Sequential()
        model.add_module('dense_0',nn.Linear(self.M, architecture[0])) 
        model.add_module('relu_0',nn.ReLU())
        if dropout is True: 
            model.add_module("dropout_0", nn.Dropout(p=dp))

        # remaining hidden layer
        for k in range(1, number_of_hidden_layers):
            model.add_module(f"dense_{k}", nn.Linear(architecture[k-1], architecture[k]))
            model.add_module(f"relu_{k}", nn.ReLU())
            if dropout is True:
                model.add_module(f"dropout{k}", nn.Dropout(p=dp))
        # final output layer
        model.add_module(f"dense_{k+1}_last", nn.Linear(architecture[k], 1))
        model.add_module(f"relu_{k+1}_last", nn.ReLU())        
        
        for m in model.modules():
            if type(m) == nn.Linear:
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu')) this is what we implemented init , below is what we ended
                nn.init.xavier_uniform_(m.weight)
#                 nn.init.kaiming_uniform_(m.weight), this is pytorch default
                m.bias.data.fill_(0.0)        

        self.model = model

    
    def forward(self, x):
        
        return self.model(x)

    
    
class MLCA_NN:

    def __init__(self, X_train, Y_train, scaler=None):
        self.M = X_train.shape[1]  # number of items
        self.X_train = X_train  # training set of bundles
        self.Y_train = Y_train  # bidder's values for the bundels in X_train
        self.X_valid = None   # test/validation set of bundles
        self.Y_valid = None  # bidder's values for the bundels in X_valid
        self.model_parameters = None  # neural network parameters
        self.model = None  # keras model, i.e., the neural network
        self.scaler = scaler  # the scaler used for initially scaling the Y_train values
        self.history = None  # return value of the model.fit() method from keras
        self.loss = None  # return value of the model.fit() method from keras
        #self.device = torch.device('cpu') # default, may be changed in init
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda')  # default, may be changed in init



        
    def initialize_model(self, model_parameters):
        
        self.model_parameters = model_parameters
        # self.device = torch.device(model_parameters['device'])
        self.model = BidderNN(self.M, model_parameters).to(self.device)
        lr = model_parameters['learning_rate']
                
        # ADAM = adaptive moment estimation a first-order gradient-based optimization algorithm
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr, betas=(0.9, 0.999), 
                                    weight_decay=0.0, amsgrad=False)
        self.criterion = nn.L1Loss(reduction='mean')
        logging.debug('Neural Net initialized')


    def __get_reg_loss(self, model):
        regularization_type = self.model_parameters['regularization_type']
        r = self.model_parameters['regularization']
        w1, w2 = 0,0
        # set regularization
        if regularization_type == 'l2' or regularization_type is None:
            w2 = r
        if regularization_type == 'l1':
            w1 = r
        if regularization_type == 'l1_l2':
            w1,w2 = r, r
        
        l1_regularization, l2_regularization = torch.FloatTensor([0]).to(self.device), torch.FloatTensor([0]).to(self.device)
                
        for name, param in model.named_parameters():
            if 'last' in name: continue
            l1_regularization += torch.sum( torch.abs(param) )
            l2_regularization += torch.sum( torch.square(param) )
        
        return w1*l1_regularization + w2*l2_regularization
        
        
    def fit(self, epochs, batch_size, X_valid=None, Y_valid=None):
        # set test set if desired
        self.X_valid = X_valid
        self.Y_valid = Y_valid

        size = self.X_train.shape[0]
        N_iter = size//batch_size + int(bool(size%batch_size))

        X = torch.FloatTensor(self.X_train).to(self.device)
        Y = torch.FloatTensor(self.Y_train).to(self.device)

        epoch_losses = {'train':[], 'val':[] ,'reg':[]}

        for n in range(epochs):

            self.model.train()
            losses = {'train':[], 'val':[] ,'reg':[]}
            nsamples = 0
            indices = np.arange(len(X)) 
            np.random.shuffle(indices)

            for i in range(N_iter):
                x = X[indices[i*batch_size: (i+1)*batch_size]]
                y = Y[indices[i*batch_size: (i+1)*batch_size]]
                nsamples += len(x)
                self.optimizer.zero_grad()
                # Compute prediction and loss
                pred = self.model(x)
                # mse_loss = self.criterion(pred.squeeze(), y)
                mse_loss = self.criterion(pred.flatten(), y.flatten())

                reg_loss = self.__get_reg_loss(self.model)
                loss = mse_loss + reg_loss
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                losses['train'].append(loss.item())
                losses['reg'].append(reg_loss.item())

            epoch_losses['train'].append(np.mean(losses['train']))

            if (self.X_valid is not None) and (self.Y_valid is not None):
                self.model.eval()
                Xval = torch.FloatTensor(self.X_valid).to(self.device)
                Yval = torch.FloatTensor(self.Y_valid).to(self.device)

                size_val = self.X_valid.shape[0]
                N_iter_val = size_val//batch_size + int(bool(size_val%batch_size))

                with torch.no_grad():
                    for i in range(N_iter_val):
                        x = Xval[i*batch_size: (i+1)*batch_size]
                        y = Yval[i*batch_size: (i+1)*batch_size]
                        pred = self.model(x)
                        mse_loss = self.criterion(pred.squeeze(), y)
                        reg_loss = self.__get_reg_loss()
                        loss = mse_loss + reg_loss
                        losses['val'].append(loss.item())

                epoch_losses['val'].append(np.mean(losses['val']))

            
#             loss = self.loss_info(batch_size, plot=False)
#         return (loss)
        tr, val = None, None
        # TO DO: val icin de en son loss'un varligini kontrol edip sadece onu cekecek
        tr_orig, val_orig = epoch_losses['train'][-1], epoch_losses['val']
        return ((tr, val, tr_orig, val_orig))


print('MLCA NN Class imported')
