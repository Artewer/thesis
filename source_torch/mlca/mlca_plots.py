import matplotlib.pyplot as plt
import scipy.spatial as scs
import numpy as np
from collections import OrderedDict


#this func is used to plot the initialized bundles

def plot_initial_bundles(initial_bids_rand):

    fig = plt.figure(figsize=(28,55))
    counter=1
    bidder_size = len(initial_bids_rand)
    for key, value in initial_bids_rand.items():
        fig.add_subplot(bidder_size, 1, counter)
        plt.imshow(initial_bids_rand[key][0])
        counter=counter+1
        plt.title(key)
    fig.show()

#works
def plot_NN_losses(configdict_mlca,output):
    ##this is used for plotting losses for each bidder from each marginal economy (need to check for marginal econ in mlca)

    bidder_ids = configdict_mlca['bidder_ids']
    total_loss_bidd = OrderedDict(list(('Bidder_{}'.format(bidder_id), []) for bidder_id in bidder_ids))

    nn_losses = output['Statistics']['NN Losses']

    for key, value in nn_losses.items():
        for keys, values in nn_losses[key].items():
            total_loss_bidd[keys].append(np.array(nn_losses[key][keys]))
    fig = plt.figure(figsize=(14, 55))
    counter = 1

    for key, value in total_loss_bidd.items():
        fig.add_subplot(7, 1, counter)
        j = total_loss_bidd[key]
        for econ in range(len(j)):
            plt.hist(j[econ][:, 2], bins=np.arange(0, 50, 0.5), alpha=0.3)
        counter = counter + 1
        plt.title(key)
    fig.show()

# def plot_convergence(configdict_mlca,result):