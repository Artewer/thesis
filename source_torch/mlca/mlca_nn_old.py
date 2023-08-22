
# Libs
import numpy as np
import logging
import matplotlib.pyplot as plt
# from tensorflow.keras import models,layers,regularizers,optimizers
import torch.nn as nn
import torch
import torch.optim as optim



# %% NN Class for each bidder


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
        self.device = torch.device('cuda')
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def initialize_model(self, model_parameters):
        self.model_parameters = model_parameters
        # model parameters is a tuple:(r=regularization_parameters,lr=learning rate for ADAM, dim=number and dimension of hidden layers, dropout=boolean if dropout is used in trainig, dp=dropout rate,epochs=epochs, batch_size=batch_size, regularization_type=regularization_type)
        lr = self.model_parameters['learning_rate']
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
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.0)        

        self.model = model
        
        # ADAM = adaptive moment estimation a first-order gradient-based optimization algorithm
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr, betas=(0.9, 0.999), weight_decay=0.0, amsgrad=False)
        self.criterion = nn.L1Loss(reduction='mean')
        logging.debug('Neural Net initialized')

        
    def __get_reg_loss(self):
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
        
        l1_regularization, l2_regularization = torch.FloatTensor([0]), torch.FloatTensor([0])
                
        for name, param in self.model.named_parameters():
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
        self.model.to(self.device)

        epoch_losses = {'train':[], 'val':[]}

        for n in range(epochs):
            self.model.train()
            losses = {'train':[], 'val':[], 'reg':[]}
            indices = np.arange(len(X)) 
            np.random.shuffle(indices)

            for i in range(N_iter):
                x = X[indices[i*batch_size: (i+1)*batch_size]]
                y = Y[indices[i*batch_size: (i+1)*batch_size]]

                self.optimizer.zero_grad()
                # Compute prediction and loss
                pred = self.model(x)
                l1_loss = self.criterion(pred.squeeze(), y)
                reg_loss = self.__get_reg_loss()
                loss = l1_loss + reg_loss

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                losses['train'].append(loss.item())
                # losses['reg'].append(reg_loss.item()*len(x))


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
                        mse_loss = self.criterion(pred, y)
                        reg_loss = self.__get_reg_loss()
                        loss = mse_loss + reg_loss
                        losses['val'].append(loss)

                epoch_losses['val'].append(np.mean(losses['val']))

            
#             loss = self.loss_info(batch_size, plot=False)
#         return (loss)
        tr, val = None, None
        # TO DO: val icin de en son loss'un varligini kontrol edip sadece onu cekecek
        tr_orig, val_orig = epoch_losses['train'][-1], epoch_losses['val']
        return ((tr, val, tr_orig, val_orig))

    def loss_info(self, batch_size, plot=True, scale=None):
        '''
        Returns
        Scalar test loss (if the model has a single output and no metrics) 
        or list of scalars (if the model has multiple outputs and/or metrics). 
        The attribute model.metrics_names will give you the display labels for the scalar outputs.
        '''
        logging.debug('Model Parameters:')
        for k,v in self.model_parameters.items():
            logging.debug(k + ': %s', v)
        tr = None
        tr_orig = None
        val = None
        val_orig = None
        # if scaler attribute was specified
        if self.scaler is not None:
            logging.debug(' ')
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            # errors on the training set
            tr = self.model.evaluate(x=self.X_train, y=self.Y_train, verbose=0)
            tr_orig = float(self.scaler.inverse_transform([[tr]]))
            if (self.X_valid is not None) and (self.Y_valid is not None):
                # errors on the test set
                val = self.model.evaluate(x=self.X_valid, y=self.Y_valid, verbose=0)
                val_orig = float(self.scaler.inverse_transform([[val]]))
        # data has not been scaled by scaler, i.e., scaler == None
        else:
            tr_orig = self.model.evaluate(x=self.X_train, y=self.Y_train, verbose=0)
            if (self.X_valid is not None) and (self.Y_valid is not None):
                val_orig = self.model.evaluate(x=self.X_valid, y=self.Y_valid, verbose=0)
        # print errors
        if tr is not None:
            logging.info('Train Error Scaled %s', tr)
        if val is not None:
            logging.info('Validation Error Scaled %s', val)
        if tr_orig is not None:
            logging.info('Train Error Orig. %s', tr_orig)
        if val_orig is not None:
            logging.info('Validation Error Orig %s', val_orig)
        logging.debug('---------------------------------------------')

        # plot results
        if plot is True:
            # recalculate predicted values for the training set and test set, which are used for the true vs. predicted plot.
            Y_hat_train = self.model.predict(x=self.X_train, batch_size=batch_size).flatten()
            if (self.X_valid is not None) and (self.Y_valid is not None):
                Y_hat_valid = self.model.predict(x=self.X_valid, batch_size=batch_size).flatten()
            fig, ax = plt.subplots(1, 2)
            plt.subplots_adjust(hspace=0.3)
            if scale == 'log':
                ax[0].set_yscale('log')
            ax[0].plot(self.history.history['loss'])
            if (self.X_valid is not None) and (self.Y_valid is not None):
                ax[0].plot(self.history.history['val_loss'])
            ax[0].set_title('Training vs. Test Loss DNN', fontsize=30)
            ax[0].set_ylabel('Mean Absolute Error', fontsize=25)
            ax[0].set_xlabel('Number of Epochs', fontsize=25)
            ax[0].legend(['Train', 'Test'], loc='upper right', fontsize=20)
            ax[1].plot(Y_hat_train, self.Y_train, 'bo')
            ax[1].set_ylabel('True Values', fontsize=25)
            ax[1].set_xlabel('Predicted Values', fontsize=25)
            ax[1].set_title('Prediction Accuracy', fontsize=30)

            if (self.X_valid is not None) and (self.Y_valid is not None):
                ax[1].plot(Y_hat_valid, self.Y_valid, 'go')
            ax[1].legend(['Training Points', 'Test Points'], loc='upper left', fontsize=20)
            lims = [
                np.min([ax[1].get_xlim(), ax[1].get_ylim()]),  # min of both axes
                np.max([ax[1].get_xlim(), ax[1].get_ylim()]),  # max of both axes
            ]
            ax[1].plot(lims, lims, 'k-')
            ax[1].set_aspect('equal')
            ax[1].set_xlim(lims)
            ax[1].set_ylim(lims)
        return((tr, val, tr_orig, val_orig))

print('MLCA NN Class imported')
