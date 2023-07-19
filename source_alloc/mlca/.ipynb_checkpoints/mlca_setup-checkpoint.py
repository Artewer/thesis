import logging
import pandas as pd
from collections import OrderedDict
import numpy as np

#### (1) Value model parameters
### TODO: number of bidders and maybe number of items should be able to controlled
def set_value_model_parameters(configdict):
    if configdict['SATS_domain_name'] == 'LSVM':
        SATS_domain_name = 'LSVM'
        N = 6  # number of bidders
        M = 18  # number of items
        bidder_types = 2
        scaler = None
        print('\n------------------------ SATS parameters ------------------------')
        print('Value Model:', SATS_domain_name)
        print('Number of Bidders: ', N)
        print('Number of BidderTypes: ', bidder_types)
        print('Number of Items: ', M)
        print('Scaler: ', scaler)
    # =============================================================================
    elif configdict['SATS_domain_name'] == 'GSVM':
        SATS_domain_name = 'GSVM'
        N = 7  # number of bidders
        M = 18  # number of items
        bidder_types = 2
        scaler = None
        print('\n------------------------ SATS parameters ------------------------')
        print('Value Model: ', SATS_domain_name)
        print('Number of Bidders: ', N)
        print('Number of BidderTypes: ', bidder_types)
        print('Number of Items: ', M)
        print('Scaler: ', scaler)
    # =============================================================================
    elif configdict['SATS_domain_name'] == 'MRVM':
        SATS_domain_name = 'MRVM'
        N = 10  # number of bidders
        M = 98  # number of items
        bidder_types = 3
        # scaler = None


        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 500))
        print('\n------------------------ SATS parameters ------------------------')
        print('\nValue Model: ', SATS_domain_name)
        print('Number of Bidders: ', N)
        print('Number of BidderTypes: ', bidder_types)
        print('Number of Items: ', M)
        print('Scaler: ', scaler)

    configdict['bidder_ids'] = list(range(0, N))
    configdict['bidders'] = N
    configdict['items'] = M
    configdict['scaler'] = scaler


#### (2) Neural Network Parameters
###TODO: number of bidders should be able to controlled

    # (2) Neural Network Parameters
    lr = configdict['lra']
    regn = configdict['regularization']
    epochs = configdict['epochs']
    batch_size = configdict['batch_size']
    regularization_type = configdict['regularization_type']  # 'l1', 'l2' or 'l1_l2'
    # national bidder LSVM: id=0, GSVM:id=6, MRVM:id=7,8,9
    regularization_N = regn
    learning_rate_N = lr
    layer_N = configdict['layer_N']
    dropout_N = True
    dropout_prob_N = 0.05
    # regional bidders LSVM: id=1-5, GSVM:id=0-5, MRVM:id=3,4,5,6
    regularization_R = regn
    learning_rate_R = lr
    layer_R = configdict['layer_R']
    dropout_R = True
    dropout_prob_R = 0.05
    # local bidders MRVM:id=0,1,2
    regularization_L = regn
    learning_rate_L = lr
    layer_L = configdict['layer_L']
    dropout_L = True
    dropout_prob_L = 0.05
    SATS_domain_name = configdict['SATS_domain_name']
    NN_parameters = {}
    bidder_ids = configdict['bidder_ids']
    if SATS_domain_name == 'LSVM':
        for bidder_id in bidder_ids:
            if bidder_id == 0:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_N),
                                                                            ('learning_rate', learning_rate_N),
                                                                            ('architecture', layer_N),
                                                                            ('dropout', dropout_N),
                                                                            ('dropout_prob', dropout_prob_N),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type)])
            else:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_R),
                                                                            ('learning_rate', learning_rate_R),
                                                                            ('architecture', layer_R),
                                                                            ('dropout', dropout_R),
                                                                            ('dropout_prob', dropout_prob_R),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type)])
    if SATS_domain_name == 'GSVM':
        for bidder_id in bidder_ids:
            if bidder_id == 6:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_N),
                                                                            ('learning_rate', learning_rate_N),
                                                                            ('architecture', layer_N),
                                                                            ('dropout', dropout_N),
                                                                            ('dropout_prob', dropout_prob_N),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type)])
            else:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_R),
                                                                            ('learning_rate', learning_rate_R),
                                                                            ('architecture', layer_R),
                                                                            ('dropout', dropout_R),
                                                                            ('dropout_prob', dropout_prob_R),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type)])
    if SATS_domain_name == 'MRVM':
        for bidder_id in bidder_ids:
            if bidder_id in [0, 1, 2]:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_L),
                                                                            ('learning_rate', learning_rate_L),
                                                                            ('architecture', layer_L),
                                                                            ('dropout', dropout_L),
                                                                            ('dropout_prob', dropout_prob_L),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type)])
            if bidder_id in [3, 4, 5, 6]:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_R),
                                                                            ('learning_rate', learning_rate_R),
                                                                            ('architecture', layer_R),
                                                                            ('dropout', dropout_R),
                                                                            ('dropout_prob', dropout_prob_R),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type)])
            if bidder_id in [7, 8, 9]:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_N),
                                                                            ('learning_rate', learning_rate_N),
                                                                            ('architecture', layer_N),
                                                                            ('dropout', dropout_N),
                                                                            ('dropout_prob', dropout_prob_N),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type)])

    print('\n------------------------ DNN  parameters ------------------------')
    print('Epochs:', epochs)
    print('Batch Size:', batch_size)
    print('Regularization:', regularization_type)
    for key in list(NN_parameters.keys()):
        print()
        print(key + ':')
        [print(k + ':', v) for k, v in NN_parameters[key].items()]

    configdict['NN_parameters'] = NN_parameters



#### (3) NN_ALLOC PARAMETERS


    # (3)NN_ALLOC PARAMETERS
# first try
#     NN_Alloc_parameters = OrderedDict([('architecture', [1500, 500, 100]),
#                                        ('learning_rate', np.arange(1e-06, 2e-06, 1e-06)),
#                                        ('number_of_networks', 1), ('sample_number', 5000),
#                                        ('epoch', 8000), ('batch_size', None)])

    NN_Alloc_parameters =configdict['NN_Alloc_parameters']



    print('\n------------------------ NN ALLOC  parameters ------------------------')
    for key,v in NN_Alloc_parameters.items():
        print(key+':', v)
    configdict['NN_Alloc_parameters'] = NN_Alloc_parameters




#### (4) MLCA  parameters
##qround is amount of marginal economies


    # (4) MLCA specific parameters
    Qinit = configdict['Qinit']
    Qmax = configdict['Qmax']
    configdict['Qround']=configdict['bidders']
    Qround= configdict['Qround']
    SATS_auction_instance_seed = configdict['SATS_auction_instance_seed']

    print('\n------------------------ MLCA  parameters ------------------------')
    print('Qinit:', Qinit)
    print('Qmax:', Qmax)
    print('Qround:', Qround)
    print('Seed SATS Instance: ', SATS_auction_instance_seed)
    #%% Start DNN-based MLCA
    return(configdict)

