# Libs
# Libs
import logging
import pandas as pd
from collections import OrderedDict
import pickle
import torch
import numpy as np

# Own Modules
from source_torch.mlca.mlca import mlca_mechanism
from source_torch.mlca.mlca_setup import set_value_model_parameters
import datetime

# #import cdist
# from scipy.spatial.distance import cdist

# # Own Modules
# from source_torch.mlca.mlca import mlca_mechanism
# from source_torch.mlca.mlca_setup import set_value_model_parameters

torch.cuda.set_device(3)
torch.cuda.is_available()

mean_scores = 0
stats = []

x = datetime.datetime.now()
m = x.month
d = x.day
h = x.hour
mi = x.minute
instance = list(range(120,150,10))




for i in range(1):
    # log debug to console
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    #logging.basicConfig(level=logging.WARNING, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')
#     logging.basicConfig(level=logging.WARNING, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s:               %(message)s', datefmt='%H:%M:%S')    
    
    configdict_mlca = OrderedDict([('SATS_domain_name','MRVM'),
                        ('SATS_auction_instance_seed', instance[i]),
                        ('bidders',0),
                        ('items',0),
                        ('bidder_ids',0),
                        ('Qinit', 10),
                        ('Qmax', 100),
                        ('Qround',0),
                        ('Starter','mlca_extra'),
                        ('epochs' , 1),
                        ('batch_size' , 32),
                        ('regularization_type' , 'l1_l2'),
                        ('layer_N' , [16, 16]),
                        ('layer_R' ,[16, 16] ),
                        ('layer_L' , [16, 16]),
                        ('NN_parameters',[]),
                        ('bigM',6000),
                        ('warm_start',False),
                        ('MIP_parameters',[]),
                        ('scaler',None),
                        ('init_bids_and_fitted_scaler',[None,None]),
                        ('return_allocation',True),
                        ('return_payments',True),
                        ('lr',0.01),
                        ('regn',1e-5),
                        ('calc_efficiency_per_iteration',True),
                        ('active_learning_algorithm', 'gali')])

    configdict_mlca = set_value_model_parameters(configdict_mlca)

    res = mlca_mechanism(configdict = configdict_mlca)
print(res)
print(scores)

print(np.mean(scores), np.std(scores))
