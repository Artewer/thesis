import logging
import pandas as pd
from collections import OrderedDict


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
    elif configdict['SATS_domain_name'] == 'SRVM':
        SATS_domain_name = 'SRVM'
        N = 7  # number of bidders
        M = 29  # number of items
        bidder_types = 4
        # scaler = None

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 500))
        print('\n------------------------ SATS parameters ------------------------')
        print('\nValue Model: ', SATS_domain_name)
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
    device = configdict['device']
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
                                                                             regularization_type),
                                                                            ('device', device)])
            else:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_R),
                                                                            ('learning_rate', learning_rate_R),
                                                                            ('architecture', layer_R),
                                                                            ('dropout', dropout_R),
                                                                            ('dropout_prob', dropout_prob_R),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type),
                                                                             ('device', device)])
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
                                                                             regularization_type),
                                                                             ('device', device)])
            else:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_R),
                                                                            ('learning_rate', learning_rate_R),
                                                                            ('architecture', layer_R),
                                                                            ('dropout', dropout_R),
                                                                            ('dropout_prob', dropout_prob_R),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type),
                                                                             ('device', device)])
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
                                                                             regularization_type),
                                                                             ('device', device)])
            if bidder_id in [3, 4, 5, 6]:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_R),
                                                                            ('learning_rate', learning_rate_R),
                                                                            ('architecture', layer_R),
                                                                            ('dropout', dropout_R),
                                                                            ('dropout_prob', dropout_prob_R),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type),
                                                                             ('device', device)])
            if bidder_id in [7, 8, 9]:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_N),
                                                                            ('learning_rate', learning_rate_N),
                                                                            ('architecture', layer_N),
                                                                            ('dropout', dropout_N),
                                                                            ('dropout_prob', dropout_prob_N),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type),
                                                                             ('device', device)])
    if SATS_domain_name == 'SRVM':
        for bidder_id in bidder_ids:
            if bidder_id in [0, 1]:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_L),
                                                                            ('learning_rate', learning_rate_L),
                                                                            ('architecture', layer_L),
                                                                            ('dropout', dropout_L),
                                                                            ('dropout_prob', dropout_prob_L),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type),
                                                                             ('device', device)])
            if bidder_id in [2, 3, 4]:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_R),
                                                                            ('learning_rate', learning_rate_R),
                                                                            ('architecture', layer_R),
                                                                            ('dropout', dropout_R),
                                                                            ('dropout_prob', dropout_prob_R),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type),
                                                                             ('device', device)])
            if bidder_id in [5, 6]:
                NN_parameters['Bidder_{}'.format(bidder_id)] = OrderedDict([('regularization', regularization_N),
                                                                            ('learning_rate', learning_rate_N),
                                                                            ('architecture', layer_N),
                                                                            ('dropout', dropout_N),
                                                                            ('dropout_prob', dropout_prob_N),
                                                                            ('epochs', epochs),
                                                                            ('batch_size', batch_size),
                                                                            ('regularization_type',
                                                                             regularization_type),
                                                                             ('device', device)])

    print('\n------------------------ DNN  parameters ------------------------')
    print('Epochs:', epochs)
    print('Batch Size:', batch_size)
    print('Regularization:', regularization_type)
    for key in list(NN_parameters.keys()):
        print()
        print(key + ':')
        [print(k + ':', v) for k, v in NN_parameters[key].items()]

    configdict['NN_parameters'] = NN_parameters



#### (3) MIP parameters



    # (3) MIP parameters
    bigM = configdict['bigM']
    Mip_bounds_tightening = 'IA'   # False ,'IA' or 'LP'
    warm_start = False # configdict['warm_start']
    time_limit = 1800 #1h = 3600sec, previously set to 1800
    relative_gap = 0.001
    integrality_tol = 1e-8
    attempts_DNN_WDP = 10000 # was 5
    MIP_parameters = OrderedDict([('bigM',bigM),('mip_bounds_tightening',Mip_bounds_tightening), ('warm_start',warm_start),
                                  ('time_limit',time_limit), ('relative_gap',relative_gap), ('integrality_tol',integrality_tol), ('attempts_DNN_WDP',attempts_DNN_WDP)])
    print('\n------------------------ MIP  parameters ------------------------')
    for key,v in MIP_parameters.items():
        print(key+':', v)
    configdict['MIP_parameters'] = MIP_parameters



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

