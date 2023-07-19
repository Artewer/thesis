import logging
import pandas as pd
from collections import OrderedDict


#### (1) Value model parameters
### TODO: number of bidders and maybe number of items should be able to controlled
def set_value_model_parameters_pvm(configdict):
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
    bidder_ids = configdict['bidder_ids']

#### (2) Neural Network Parameters
###TODO: number of bidders should be able to controlled

    # (2) Neural Network Parameters
    epochs = configdict['epochs']
    batch_size = configdict['batch_size']
    regularization_type = configdict['regularization_type']  # 'l1', 'l2' or 'l1_l2'
    # national bidder LSVM: id=0, GSVM:id=6, MRVM:id=7,8,9
    regularization_N = 1e-5
    learning_rate_N = 0.01
    layer_N = configdict['layer_N']
    dropout_N = True
    dropout_prob_N = 0.05
    # regional bidders LSVM: id=1-5, GSVM:id=0-5, MRVM:id=3,4,5,6
    regularization_R = 1e-5
    learning_rate_R = 0.01
    layer_R = configdict['layer_R']
    dropout_R = True
    dropout_prob_R = 0.05
    # local bidders MRVM:id=0,1,2
    regularization_L = 1e-5
    learning_rate_L = 0.01
    layer_L = configdict['layer_L']
    dropout_L = True
    dropout_prob_L = 0.05
    model_name = configdict['SATS_domain_name']
    DNN_parameters = {}
    if model_name == 'LSVM':
        for bidder_id in bidder_ids:
            if bidder_id == 0:
                DNN_parameters['Bidder_{}'.format(bidder_id)] = (
                regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
            else:
                DNN_parameters['Bidder_{}'.format(bidder_id)] = (
                regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
    if model_name == 'GSVM':
        for bidder_id in bidder_ids:
            if bidder_id == 6:
                DNN_parameters['Bidder_{}'.format(bidder_id)] = (
                regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
            else:
                DNN_parameters['Bidder_{}'.format(bidder_id)] = (
                regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
    if model_name == 'MRVM':
        for bidder_id in bidder_ids:
            if bidder_id in [0, 1, 2]:
                DNN_parameters['Bidder_{}'.format(bidder_id)] = (
                regularization_L, learning_rate_L, layer_L, dropout_L, dropout_prob_L)
            if bidder_id in [3, 4, 5, 6]:
                DNN_parameters['Bidder_{}'.format(bidder_id)] = (
                regularization_R, learning_rate_R, layer_R, dropout_R, dropout_prob_R)
            if bidder_id in [7, 8, 9]:
                DNN_parameters['Bidder_{}'.format(bidder_id)] = (
                regularization_N, learning_rate_N, layer_N, dropout_N, dropout_prob_N)
    sample_weight_on = False
    sample_weight_scaling = None

    print('\n------------------------ DNN  parameters ------------------------')
    print('Epochs:', epochs)
    print('Batch Size:', batch_size)
    print('Regularization:', regularization_type)
    for key in list(DNN_parameters.keys()):
        print(key, DNN_parameters[key])
    print('Sample weighting:', sample_weight_on)
    print('Sample weight scaling:', sample_weight_scaling)

    configdict['NN_parameters'] = DNN_parameters



#### (3) MIP parameters



    # (3) MIP parameters
    L = configdict['L']
    Mip_bounds_tightening = 'IA'   # False ,'IA' or 'LP'
    warm_start = configdict['warm_start']
    time_limit = 1800  #1h = 3600sec
    relative_gap = 0.001
    integrality_tol = 1e-8
    attempts_DNN_WDP = 5
    MIP_parameters = OrderedDict([('L',L),('mip_bounds_tightening',Mip_bounds_tightening), ('warm_start',warm_start),
                                  ('time_limit',time_limit), ('relative_gap',relative_gap), ('integrality_tol',integrality_tol), ('attempts_DNN_WDP',attempts_DNN_WDP)])
    print('\n------------------------ MIP  parameters ------------------------')
    for key,v in MIP_parameters.items():
        print(key+':', v)
    configdict['MIP_parameters'] = MIP_parameters



#### (4) MLCA  parameters
##qround is amount of marginal economies

    # (4) PVM specific parameters
    caps = configdict['caps']  # [c_0, c_e] with initial bids c0 and maximal number of value queries ce
    seed_instance = configdict['SATS_auction_instance_seed']
    min_iteration = 1
    print('\n------------------------ PVM  parameters ------------------------')
    print(caps)
    print('Seed: ', seed_instance)
    print('min_iteration:', min_iteration)
    # %% Start DNN-based PVM

    return(configdict)
