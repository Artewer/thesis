# Libs
import numpy as np
import logging
import matplotlib.pyplot as plt

from collections import OrderedDict

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

#########################################this module goes to the direction of MAX SW estimated##########
class Alloc_Soft_MAX(nn.Module):
    def __init__(self, input_parameters_alloc, model_parameters_alloc):
        super(Alloc_Soft_MAX, self).__init__()

        self.model_parameters_alloc = model_parameters_alloc  ##gets the model parameters
        self.input_parameters_alloc = input_parameters_alloc  # gets the model input

        architecture = self.model_parameters_alloc['architecture']

        self.weight_dim = self.input_parameters_alloc['weight_dim']
        self.bidder_num = self.input_parameters_alloc['bidder_num']
        self.item_num = self.input_parameters_alloc['item_num']

        architecture = [int(layer) for layer in architecture]  # integer check
        number_of_hidden_layers = len(architecture)

        self.first = nn.Linear(self.weight_dim * self.bidder_num, architecture[0])
        self.second = nn.Linear(architecture[0], architecture[1])
        self.last = nn.Linear(architecture[1], self.item_num * self.bidder_num)


    def forward(self, x):

        x = F.relu(self.first(x))
        x = F.relu(self.second(x))
        x = F.relu(self.last(x))
        prediction = x.view(-1, self.item_num)
        x = F.softmax(prediction, dim=0)

        return x


class MLCA_NN_ALLOC_MAX:

    def __init__(self, models, scaler=None):
        # self.M = models[list(models.keys())[0]].model[0].weight.data.T.numpy().shape[0] # input
        self.M = models[list(models.keys())[0]].model.state_dict()  # all weigts of 1 bidder
        self.item_num = self.M['dense_0.weight'].shape[1]  # item number
        self.Models = models  # dict of pytorch models of valuation function
        self.Sol = None
        # sorted list of bidders
        self.sorted_bidders = list(self.Models.keys())
        self.sorted_bidders.sort()
        self.bidder_num = len(models)  # number of bidders

        self.model_parameters_alloc = None  # neural network parameters
        self.input_parameters_alloc = None
        self.weights_dim = None

        self.scaler = scaler  # the scaler used for initially scaling the Y_train values
        self.device = torch.device('cpu')  # default, may be changed in init

        self.weights_dict = OrderedDict()
        self.input_to_alloc = torch.empty([0]).to(self.device)

        for k, net in self.Models.items():
            weights = torch.cat([param.clone().view(-1) for param in net.model.parameters()]).detach()
            self.weights_dict[k] = weights
            self.input_to_alloc = torch.cat((self.input_to_alloc, weights))
        self.weight_dim = len(weights)
        self.input_parameters_alloc = OrderedDict([('weight_dim', self.weight_dim), ('bidder_num', self.bidder_num)
                                                      , ('item_num', self.item_num)])
        self.best_sw_over_all = 0
        self.max_sampled_allocation_over_all = []

    #     def _initialize_model_alloc(self, model_parameters_alloc):

    #         self.model_parameters_alloc = model_parameters_alloc

    #         # self.device = torch.device(model_parameters['device'])
    # #         self.alloc_soft = Alloc_Soft2(self.input_parameters_alloc,self.model_parameters_alloc).to(self.device)
    # #         lr = self.model_parameters_alloc['learning_rate']

    #         # ADAM = adaptive moment estimation a first-order gradient-based optimization algorithm
    # #         self.optimizer_alloc = optim.Adam(self.alloc_soft.parameters(),lr=lr, betas=(0.9, 0.999),
    # #                                     weight_decay=0.0, amsgrad=False)
    # #         criterion_bce = nn.BCELoss(reduction='sum')

    #         logging.debug('Neural Net initialized')

    def fit_alloc(self, model_parameters_alloc):

        self.model_parameters_alloc = model_parameters_alloc

        lra = self.model_parameters_alloc['learning_rate']
        number_of_networks = self.model_parameters_alloc['number_of_networks']
        probability_sample_no = self.model_parameters_alloc['sample_number']

        for k, net in self.Models.items():
            net.model.train(False)


        for lr in lra:
            # print("FOR LEARNING RATE \t", lr)
            alloc_loss_total = 0
            for j in range(number_of_networks):

                self.alloc_soft = Alloc_Soft_MAX(self.input_parameters_alloc, self.model_parameters_alloc).to(self.device)

                self.optimizer_alloc = optim.Adam(self.alloc_soft.parameters(), lr=lr, betas=(0.9, 0.999),
                                                  weight_decay=0.0, amsgrad=False)
                self.alloc_soft.train()
                criterion_bce = nn.BCELoss(reduction='sum')

                epoch_size = self.model_parameters_alloc['epoch']
                loss = 0
                last_epoch_change = 0
                best_sw = 0
                for epoch in range(epoch_size):
                    self.optimizer_alloc.zero_grad()
                    prob = self.alloc_soft(self.input_to_alloc)
                    sampled_allocation_indices = torch.multinomial(torch.transpose(prob, 0, 1),
                                                                   probability_sample_no, replacement=True)
                    onehot = nn.functional.one_hot(sampled_allocation_indices)
                    sampled_allocations = torch.transpose(onehot, 0, 2).to(dtype=torch.float32)
                    summa_value = 0

                    # one of the idea was to sample one bundle and to optimization
                    # the other idea is to just sample and find a expected sw then get the
                    # argmax and move towards that direction

                    with torch.no_grad():
                        for i, (network_name, network) in enumerate(self.Models.items()):
                            net_out = network.model(sampled_allocations[i])
                            summa_value = summa_value + net_out.detach()
                        max_sample_indice = torch.argmax(summa_value)
                        best_sw_epoch = summa_value[max_sample_indice]
                        if best_sw_epoch > best_sw:
                            max_sampled_allocation = sampled_allocations[:, max_sample_indice, :].flatten()
                            best_sw = best_sw_epoch
                            last_epoch_change = epoch
                        if best_sw_epoch > self.best_sw_over_all:
                            self.max_sampled_allocation_over_all = sampled_allocations[:, max_sample_indice,
                                                                   :].flatten()
                            self.best_sw_over_all = best_sw_epoch

                    #     sparsity_loss = torch.norm(prediction,dim=(0),p=1)
                    #     loss = criterion(summa_value, max_value) + sparsity_loss.sum()
                    #     loss = -summa_value

                    loss = criterion_bce(prob.flatten(), max_sampled_allocation)
                    loss.backward()
                    self.optimizer_alloc.step()
                    # if epoch % 10 == 0:
                    #     print(f"At epoch {epoch}, loss.item is {loss.item()}, best sw {best_sw.item()}")
                alloc_loss_total += loss.item()
                # print(f"For network {j}, last epoch change at {last_epoch_change}, best sw {best_sw}")
            avg_sw = alloc_loss_total / number_of_networks

        self.Sol = self.max_sampled_allocation_over_all.view(-1,self.item_num).cpu().detach().numpy()
        return(self.Sol)

print('MLCA NN ALLOC MAX Class imported')