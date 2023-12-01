# Libs
import logging
import pandas as pd
from collections import OrderedDict
import torch

# Own Modules
from source_torch.mlca.mlca import mlca_mechanism
from source_torch.mlca.mlca_setup import set_value_model_parameters

import datetime
import os
from source_torch.util import save_result, load_result
import numpy as np
import os

torch.cuda.set_device(2)
device = torch.device("cuda:2")
iterations = 20

seed = 2
while iterations > 0:
    x = datetime.datetime.now()
    m = x.month
    d = x.day
    h = x.hour
    mi = x.minute

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # log debug to console
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')
    #logging.basicConfig(level=logging.WARNING, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')    
    
    configdict_mlca = OrderedDict([('SATS_domain_name','GSVM'),
                        ('SATS_auction_instance_seed', seed),
                        ('bidders',0),
                        ('items',0),
                        ('bidder_ids',0),
                        ('Qinit', 30), 
                        ('Qmax', 50),#50
                        ('Qround',0),
                        ('Starter','mlca_extra'),
                        ('epochs' , 512),#512
                        ('batch_size' , 32),
                        ('regularization_type' , 'l1_l2'),
                        ('layer_N' , [10, 10]),
                        ('layer_R' ,[32, 32] ),
                        ('layer_L' , [16, 16]),
                        ('device', device),
                        ('NN_parameters', []),
                        ('bigM',6000),
                        ('warm_start',False),
                        ('MIP_parameters',[]),
                        ('scaler',None),
                        ('init_bids_and_fitted_scaler',[None,None]),
                        ('return_allocation',True),
                        ('return_payments',True),
                        ('lra',0.01),
                        ('regularization',1e-5),
                        ('calc_efficiency_per_iteration',True),
                        ('active_learning_algorithm', 'unif'),
                        ('presampled_n', 20),
                        ('presampled_algorithm', 'unif')])

    configdict_mlca = set_value_model_parameters(configdict_mlca)

    res = mlca_mechanism(configdict = configdict_mlca)
    seed += 1

    if res == 'UNIQUENESS CHECK FAILED see logfile':
        # add one more iteration to the loop
        iterations += 1
        #continue

    iterations -= 1

    dirname = './experiments/MLCA/Torch/results/' + configdict_mlca['SATS_domain_name'] +'_'+ configdict_mlca['active_learning_algorithm']
    result_dir = './experiments/MLCA/Torch/results/' + configdict_mlca['SATS_domain_name'] +'_' + configdict_mlca['active_learning_algorithm'] +'/' +str(m)+'_'+str(d) + '_' + str(h) + '_' + str(mi) + '_' + str(configdict_mlca['SATS_auction_instance_seed'])
    os.makedirs(dirname,exist_ok=True)
    save_result(result_dir, res)