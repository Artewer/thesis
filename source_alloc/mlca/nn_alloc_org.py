#

# Libs
import numpy as np
import logging
import matplotlib.pyplot as plt

from collections import OrderedDict

import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


"""
FILE DESCRIPTION:
This file implements the class Alloc_Soft (Allocation Neural Network) and MLCA_NN_ALLOC.
The class Alloc_Soft is a subclass the nn.Module, the basis of all neural network modules, receives the model parameters and sets the appropriate values for each layer of the Deep Neural Network. It serves as a constructor of a network that is invoked by the class MLCA_NN_ALLOC.
The class MLCA_NN_ALLOC implements the Deep Neural Network responsible for producing an optimal candidate allocation, after receiving input from the Bidder Networks containing elicited bidder valuations.
The class Alloc_Soft has following functionalities:
    0.CONSTRUCTOR: __init__(self, input_parameters_alloc, model_parameters_alloc)
        input_parameters_alloc = Parameters specifying input model, includes weight_dim (weight dimensions), bidder_num (number of bidders), and item_num (number of items)
        model_parameters_alloc = Parameters specifying allocation model, includes model architecture (as array of integers representing layers)
    1.METHOD: forward(self, x)
        This method defines the computation performed at every call. 

The class MLCA_NN_ALLOC has the following funtionalities:
    0. CONSTRUCTOR: __init__(self, models, scaler)
        models = Ordered dict of DNNs corresponding to each economy
        scaler =  Scaler instance from sklearn.preprocessing.*, used to scale training variables before creating an NN instance.
    1. METHOD: fit_alloc(self, model_parameters_alloc)
        model_parameters_alloc = Parameters of allocation model, as defined and set on subclass Alloc_Soft. In this function, parameters learning_rate, sample_number and epoch are also set. 
        This method receives as input the model parameters and initializes the model based on the Alloc_Soft class, and for each epoch it sets up and runs the Reinforce method.

"""

def unique_inds(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

class Alloc_Soft(nn.Module):
    def __init__(self, input_parameters_alloc, model_parameters_alloc):
        super(Alloc_Soft, self).__init__()

        self.model_parameters_alloc = model_parameters_alloc  ##gets the model parameters
        self.input_parameters_alloc = input_parameters_alloc  # gets the model input

        architecture = self.model_parameters_alloc['architecture'] #architecture ie:[10,10,10]

        self.weight_dim = self.input_parameters_alloc['weight_dim'] #number of parameters in bidder net
        self.bidder_num = self.input_parameters_alloc['bidder_num'] #number of bidders
        self.item_num = self.input_parameters_alloc['item_num'] #number of items

        architecture = [int(layer) for layer in architecture]  # integer check
        number_of_hidden_layers = len(architecture) #not used right now

        self.first = nn.Linear(self.weight_dim * self.bidder_num, architecture[0]) #input layer
        self.second = nn.Linear(architecture[0], architecture[1]) #1st hidden layer
        self.third = nn.Linear(architecture[1], architecture[2]) #2nd hidden layer


        self.last = nn.Linear(architecture[2], self.item_num * self.bidder_num) #output layer


    def forward(self, x):
        x = F.elu(self.first(x))
        x = F.elu(self.second(x))
        x = F.elu(self.third(x))
        x = F.elu(self.last(x))
        prediction = x.view(-1, self.item_num)
        x = F.softmax(prediction, dim=0) #softmax output

        return x

class MLCA_NN_ALLOC:

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
        self.input_parameters_alloc = None  # now empty we will fill this
        self.weights_dim = None

        self.scaler = scaler  # the scaler used for initially scaling the Y_train values
        self.device = torch.device('cuda')  # default, may be changed in init

        self.weights_dict = OrderedDict()
        self.input_to_alloc = torch.empty([0]).to(self.device)

        for k, net in self.Models.items(): #creates the input for alloc with designated size
            weights = torch.cat([param.clone().view(-1) for param in net.model.parameters()]).detach().to(self.device)
            self.weights_dict[k] = weights
            self.input_to_alloc = torch.cat((self.input_to_alloc, weights))

        self.weight_dim = len(weights)

        self.input_parameters_alloc = OrderedDict([('weight_dim', self.weight_dim), ('bidder_num', self.bidder_num)
                                                      , ('item_num', self.item_num)])

        self.avg_sampled_allocation_over_all = []
        self.best_avg_sw_over_all = 0

    def fit_alloc(self, baseline, model_parameters_alloc, R_union_S):

        self.model_parameters_alloc = model_parameters_alloc
        
        lra = self.model_parameters_alloc['learning_rate'] #learning rate is set
        probability_sample_no = self.model_parameters_alloc['sample_number'] # number of samples

        for k, net in self.Models.items():
            net.model.train(False)

        alloc_loss_total = 0

        self.alloc_soft = Alloc_Soft(self.input_parameters_alloc, self.model_parameters_alloc).to(
            self.device) #model is formed
        self.optimizer_alloc = optim.AdamW(self.alloc_soft.parameters(), lr=lra, betas=(0.9, 0.999)) #optimizer
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_alloc, 2, eta_min=3e-06,
                                                              last_epoch=-1, verbose=False) #learning rate scheduler
        self.alloc_soft.train() #alloc_net init

        epoch_size = self.model_parameters_alloc['epoch'] #number of epoch
        loss = 0
        current_sample_max = 0
        epoch_count = 0
        epoch_id = 0
        random_avg = 0

        sample_maxs = []
        this_max = []
        epoch_avgs = []
        epochs = []
        indx = 0

        best_sw = torch.zeros((self.bidder_num,))
        best_query = torch.zeros((self.bidder_num, self.item_num))
        
        for epoch in range(epoch_size):

            self.optimizer_alloc.zero_grad()

            self.prob = self.alloc_soft(self.input_to_alloc) # output of allocation net (softmax)
            # HERE RANDOM SEARCH 
            # unif_probs = [ [1/ self.bidder_num]*self.bidder_num for i in range(self.item_num)]
            # unif_dist = torch.tensor(unif_probs)
            # unif_dist = unif_dist.cuda()
            # self.dist = Categorical(probs = unif_dist) #categorical dist is set
            # AND ENDS HERE
            self.dist = Categorical(probs=self.prob.transpose(0, 1)) #categorical dist is set
            self.action = self.dist.sample(sample_shape=torch.Size([probability_sample_no])) #winners of items are sampled
            onehot = nn.functional.one_hot(self.action, self.bidder_num)
            sampled_allocations = torch.transpose(onehot, 1, 2).to(dtype=torch.float32)
            sampled_allocations = torch.transpose(sampled_allocations, 0, 1) #sampled feasible allocation instances. [bidder, prob sample, item]
        
            self.summa_value = 0
            self.avg_sw_epoch = 0

            with torch.no_grad():#get estimated valuations for each sample

                for i, (network_name, network) in enumerate(self.Models.items()):
                    # network = network.to(self.device)
                    net_out = network(sampled_allocations[i])
                    self.summa_value = self.summa_value + net_out.detach()

    
                for bidder in range(0, sampled_allocations.shape[0]):
                    if list(self.Models.keys())[bidder] in R_union_S.keys():
                        sampled_allocations_bidder = sampled_allocations[bidder]
                        sampled_allocations_bidder_unique, inds = unique_inds(sampled_allocations_bidder, dim=0)
                        sw_unique = self.summa_value[inds].squeeze()
                        elicited_bids_bidder = R_union_S[list(self.Models.keys())[bidder]]    # [number_of_queries = 50, items]
                        sw_unique_sorted, topk_inds = torch.topk(sw_unique, min(elicited_bids_bidder.shape[0]+1, sw_unique.shape[0]))
                        sampled_allocations_bidder_unique_sorted = sampled_allocations_bidder_unique[topk_inds] # 51*no_items
                        nonelicited = torch.logical_not((sampled_allocations_bidder_unique_sorted.unsqueeze(1) == elicited_bids_bidder).all(-1)).all(-1) # == has dimensions 51*50*no_items > output is 51 (size)
                        sampled_allocations_bidder_nonelicited = sampled_allocations_bidder_unique_sorted[nonelicited]
                        sw_nonelicited = sw_unique_sorted[nonelicited]
                        best_sw_temp, best_sw_ind = torch.max(sw_nonelicited, 0)
                        best_query_bidder_temp = sampled_allocations_bidder_nonelicited[best_sw_ind]
                        if (best_sw_temp > best_sw[bidder]):
                            best_query[bidder] = best_query_bidder_temp

                # improvement = torch.max(self.summa_value) - current_sample_max
                # if (improvement <= 0.03*current_sample_max):
                #     if ((epoch_id == 0) or (epoch_id == epoch-1)):
                #         epoch_id = epoch
                #         epoch_count += 1
                #     if (epoch_count == 50):
                #         print(f'Stopped after {epoch} epochs.')
                #         return best_query

                # EARLY STOPPING HAPPENS HERE
                if (torch.max(self.summa_value) > current_sample_max):
                    current_sample_max = torch.max(self.summa_value)
                    epoch_count = 0
                else:
                    epoch_count += 1

                # if (epoch_count >= 50):
                #     return best_query
                # EARLY STOPPING ENDS HERE

                    # sample_maxs.append(current_sample_max)
                    # epoch_id = 0
                    # indx += 1
                    # best_query = sampled_allocations[:,torch.argmax(self.summa_value),:]

               # random_avg = torch.mean(self.summa_value)
    
            self.avg_sw_epoch = self.summa_value.sum() / probability_sample_no # average sw for each allocation
            self.score = 0
            # GRADIENT COMPUTATION STARTS HERE
            self.log_probs = self.dist.log_prob(self.action) #log probability of each allocation
            
            self.based = self.summa_value
                    
            # self.normalized = self.based / self.based.abs().mean()

            self.scores = self.log_probs * self.based

            # self.scores=X.log_probs*self.normalized

            self.reinforce = self.scores.sum(dim=1).mean()

            loss = -self.reinforce #loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.alloc_soft.parameters(), 100,'inf')
            self.optimizer_alloc.step()

            # HERE ENDS GRADIENT COMPUTATION


            # if (epoch >= 1):
            #     improve = self.avg_sw_epoch - current_sample_max
            # else:
            #     improve = 0

            # if ((epoch >= 1) and (improve <= 0.03*current_sample_max)):
            #     if ((epoch_id == 0) or (epoch_id == epoch-1)):
            #         epoch_id = epoch
            #         epoch_count += 1
            #     if (epoch_count == 50):
            #         print(f'Exiting after {epoch} epochs.')
            #         return best_query


            if epoch % 50 == 0:
                print(f"At epoch {epoch}, avg sw {self.avg_sw_epoch} - and  current max {current_sample_max}")
            #   print(f"Max welfare out of sampled allocations at this epoch is {current_sample_max}")
                epoch_avgs.append(self.avg_sw_epoch)
                sample_maxs.append(current_sample_max)
                epochs.append(epoch)

            if epoch %10== 0 and not epoch == 0:
                self.scheduler.step()


        alloc_loss_total += loss.item()

        # plt.plot(epochs, epoch_avgs, color = 'blue', label = 'Social Welfare of Epoch')
        # plt.plot(epochs, sample_maxs, color = 'red', label = 'Max of Sampled Allocations')
        # plt.title(f'Sample Number is {probability_sample_no}')
        # plt.xlabel('Epochs')
        # plt.ylabel('Objective')
        # plt.legend()
        # plt.show()

        # return (sampled_allocations[:, 0, :])
        return best_query

print('MLCA NN ALLOC Class imported')
##########################################                   logging needs to be added                   ##############

#
# def fit_alloc(self, model_parameters_alloc):
#
#     self.model_parameters_alloc = model_parameters_alloc
#
#     lra = self.model_parameters_alloc['learning_rate']
#     number_of_networks = self.model_parameters_alloc['number_of_networks']
#     probability_sample_no = self.model_parameters_alloc['sample_number']
#
#     for k, net in self.Models.items():
#         net.model.train(False)
#
#     # criterion_bce = nn.BCELoss(reduction='sum')
#     #
#     # tolerance = np.arange(0.00001, 0.1, 0.00001)
#     #
#
#     for lr in lra:
#         print("FOR LEARNING RATE \t", lr)
#
#         alloc_loss_total = 0
#
#         for j in range(number_of_networks):
#
#             self.alloc_soft = Alloc_Soft(self.input_parameters_alloc, self.model_parameters_alloc).to(self.device)
#
#             self.optimizer_alloc = optim.Adam(self.alloc_soft.parameters(), lr=lr, betas=(0.9, 0.999),
#                                               weight_decay=0.0, amsgrad=False)
#             #                 self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer_alloc,0.5,verbose=True)
#
#             self.alloc_soft.train()
#
#             epoch_size = self.model_parameters_alloc['epoch']
#             loss = 0
#             last_epoch_change = 0
#             self.best_avg_sw = 0
#
#             for epoch in range(epoch_size):
#                 self.optimizer_alloc.zero_grad()
#                 prob = self.alloc_soft(self.input_to_alloc)
#                 sampled_allocation_indices = torch.multinomial(torch.transpose(prob, 0, 1),
#                                                                probability_sample_no, replacement=True)
#                 onehot = nn.functional.one_hot(sampled_allocation_indices)
#                 sampled_allocations = torch.transpose(onehot, 0, 2).to(dtype=torch.float32)
#                 summa_value = 0
#
#                 with torch.no_grad():
#                     for i, (network_name, network) in enumerate(self.Models.items()):
#                         # network.to(self.device)
#                         net_out = network(sampled_allocations[i])
#                         summa_value = summa_value + net_out.detach()
#
#                     avg_sw_epoch = summa_value.sum() / probability_sample_no
#
#                     to_go_plus = avg_sw_epoch + avg_sw_epoch * tolerance
#                     to_go_minus = avg_sw_epoch - avg_sw_epoch * tolerance
#
#                     for i in range(len(tolerance)):
#                         avg_indices = np.where((summa_value > to_go_minus[i]) & (summa_value < to_go_plus[i]))
#                         if avg_indices[0].size != 0:
#                             break
#
#                     avg_random_indice = np.random.choice(avg_indices[0].size)
#                     # avg_random_indice = torch.argmin(torch.abs(summa_value-avg_sw_epoch))
#                     # avg_sw_epoch = summa_value[avg_random_indice]
#
#                     # avg_sampled_allocation = sampled_allocations[:, avg_random_indice, :].flatten()
#
#                     if avg_sw_epoch > self.best_avg_sw:
#                         self.avg_sampled_allocation = sampled_allocations[:, avg_random_indice, :].flatten()
#                         self.best_avg_sw = avg_sw_epoch
#                         last_epoch_change = epoch
#
#                     if avg_sw_epoch > self.best_avg_sw_over_all:
#                         self.avg_sampled_allocation_over_all = sampled_allocations[:, avg_random_indice, :]
#
#
#                 loss = criterion_bce(prob.flatten(), self.avg_sampled_allocation)
#                 loss.backward()
#                 self.optimizer_alloc.step()
#                 if epoch % 100 == 0:
#                     print(f"At epoch {epoch}, loss.item is {loss.item()}, best sw {self.best_avg_sw.item()}")
#                 #                     if epoch %5000 == 0 and not epoch == 0:
#                 #                         self.scheduler.step()
#
#             alloc_loss_total += loss.item()
#             print(f"For network {j}, last epoch change at {last_epoch_change}, best sw {self.best_avg_sw}")
#
#     self.Sol = self.avg_sampled_allocation_over_all.view(-1,self.item_num).cpu().detach().numpy()
#     return(self.Sol)