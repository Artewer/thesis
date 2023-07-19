
# Libs
import os
import itertools
import sys
import numpy as np
import random
import time
from collections import OrderedDict
import logging
# CPLEX: Here, DOcplex is used for solving the deep neural network-based Winner Determination Problem.
import docplex
# documentation
from copy import deepcopy
from scipy.special import softmax
# Own Modules
import source_alloc.util as util
from source_alloc.mlca.mlca_nn import MLCA_NN
from source_alloc.wdp import WDP

from source_alloc.mlca.mlca_nn_alloc import MLCA_NN_ALLOC
# from source_alloc.mlca.nn_alloc_org import MLCA_NN_ALLOC
import torch


# %%


class MLCA_Economies:

    def __init__(self, SATS_auction_instance, SATS_auction_instance_seed, Qinit, Qmax, Qround, scaler):

        # STATIC ATTRIBUTES
        self.SATS_auction_instance = SATS_auction_instance  # auction instance from SATS: LSVM, GSVM, SRVM or MRVM generated via PySats.py.
        self.SATS_auction_instance_allocation = None  # true efficient allocation of auction instance
        self.SATS_auction_instance_scw = None  # social welfare of true efficient allocation of auction instance
        self.SATS_auction_instance_seed = SATS_auction_instance_seed # auction instance seed from SATS
        self.bidder_ids = list(SATS_auction_instance.get_bidder_ids())  # bidder ids in this auction instance.
        self.bidder_names = list('Bidder_{}'.format(bidder_id) for bidder_id in self.bidder_ids)
        self.N = len(self.bidder_ids)  # number of bidders
        self.good_ids = set(SATS_auction_instance.get_good_ids())  # good ids in this auction instance
        self.M = len(self.good_ids)  # number of items
        self.Qinit = Qinit  # number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure, different per bidder)
        self.Qmax = Qmax  # maximal number of possible value queries in the preference elicitation algorithm (PEA) per bidder
        self.Qround = Qround  # maximal number of marginal economies per auction round per bidder
        self.scaler = scaler  # scaler is a instance from sklearn.MinMaxScaler() used for scaling the values of the bundle-value pairs
        self.fitted_scaler = None  # fitted scaler to the initial bids
        self.mlca_allocation = None  # mlca allocation
        self.mlca_scw = None # true social welfare of mlca allocation
        self.mlca_allocation_efficiency = None # efficiency of mlca allocation
        self.NN_Alloc_parameters = None  # MIP parameters
        self.mlca_iteration = 0 # mlca iteration tracker
        subsets = list(map(list, itertools.combinations(self.bidder_ids, self.N-1))) # orderedDict containing all economies and the corresponding bidder ids of the bidders which are active in these economies.
        subsets.sort(reverse=True) #sort s.t. Marginal Economy (-0)-> Marginal Economy (-1) ->...
        # Comment out marginal economies
        self.economies = OrderedDict(list(('Marginal Economy -({})'.format(i), econ) for econ, i in zip(subsets, [[x for x in self.bidder_ids if x not in subset][0] for subset in subsets])))
        self.economies['Main Economy'] = self.bidder_ids
        self.economies_names = OrderedDict(list((key, ['Bidder_{}'.format(s) for s in value]) for key, value in self.economies.items())) # orderedDict containing all economies and the corresponding bidder names (as strings) of the bidders which are active in these economies.
        self.efficiency_per_iteration = OrderedDict()  #storage for efficiency stat per auction round
        self.welfare_per_iteration = OrderedDict()  #storage for social welfare stat per auction round
        self.efficient_allocation_per_iteration = OrderedDict()  # storage for efficent allocation per auction round

        self.exp_avg_rule = OrderedDict() # Trying out exponential averaging rule
        self.gamma = 0.6 # Rule parameter
        self.baseline = 0 # We run the MIP for a few seconds, then use the computed social welfare as the baseline for this iteration
        self.time_limit = 90000000000 # MIP timeout limit, ranging between full solution and a few seconds

        # DYNAMIC PER ECONOMY
        self.economy_status = OrderedDict(list((key, False) for key, value in self.economies.items()))  # boolean, status of economy: if already calculated.
        self.mlca_marginal_allocations = OrderedDict(list((key, None) for key, value in self.economies.items() if key!='Main Economy'))  # Allocation of the WDP based on the elicited bids
        self.mlca_marginal_scws = OrderedDict(list((key, None) for key, value in self.economies.items() if key!='Main Economy'))  # Social Welfare of the Allocation of the WDP based on the elicited bids
        self.elapsed_time_mip = OrderedDict(list((key, []) for key, value in self.economies.items()))  # stored MIP solving times per economy
        self.warm_start_sol = OrderedDict(list((key, None) for key, value in self.economies.items()))  # MIP SolveSolution object used as warm start per economy

        # DYNAMIC PER BIDDER
        self.mlca_payments = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))  # VCG-style payments in MLCA, calculated at the end
        self.elicited_bids = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))  # R=(R_1,...,R_n) elicited bids per bidder
        self.current_query_profile = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids)) # S=(S_1,...,S_n) number of actual value queries, that takes into account if the same bundle was queried from a bidder in two different economies that it is not counted twice
        self.NN_parameters = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))  # DNNs parameters as in the Class NN described.
        self.initial_bundles = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))  # elicited bids in initial phase

        # DYNAMIC PER ECONOMY & BIDDER
        self.argmax_allocation = OrderedDict(list((key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in self.economies_names.items()))  # [a,a_restr] a: argmax bundles per bidder and economy, a_restr: restricted argmax bundles per bidder and economy
        self.NN_models = OrderedDict(list((key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in self.economies_names.items()))  # tf.keras NN models
        self.losses = OrderedDict(list((key, OrderedDict(list((bidder_id, []) for bidder_id in value))) for key, value in self.economies_names.items()))  # Storage for the MAE loss during training of the DNNs

        self.nonunique_count = 0
        self.unique_count = 0
        self.nonunique_percent = 0


    def get_info(self, final_summary=False):
        if not final_summary: logging.warning('INFO')
        if final_summary: logging.warning('SUMMARY')
        logging.warning('-----------------------------------------------')
        logging.warning('Seed Auction Instance: %s', self.SATS_auction_instance_seed)
        logging.warning('Iteration of MLCA: %s', self.mlca_iteration)
        logging.warning('Number of Elicited Bids:')
        for k,v in self.elicited_bids.items():
            logging.warning(k+': %s', v[0].shape[0]-1)  # -1 because of added zero bundle
        logging.warning('Qinit: %s | Qround: %s | Qmax: %s', self.Qinit, self.Qround, self.Qmax)
        if not final_summary: logging.warning('Efficiency and welfare given elicited bids from iteration 0-%s: %s, %s\n', self.mlca_iteration-1,self.efficiency_per_iteration[self.mlca_iteration-1], self.welfare_per_iteration[self.mlca_iteration-1])

    def get_number_of_elicited_bids(self, bidder=None):
        if bidder is None:
            return OrderedDict((bidder,self.elicited_bids[bidder][0].shape[0]-1) for bidder in self.bidder_names)  # -1, because null bundle was added
        else:
            return self.elicited_bids[bidder][0].shape[0]-1

    
    def run_mip_on_elicited(self, time_limit):
        logging.debug('')
        logging.debug('Run MIP for a few seconds:')
        mip_allocation, mip_objective = self.solve_WDP(self.time_limit, self.elicited_bids, verbose=1)     
        self.baseline = mip_objective  


    def calculate_efficiency_welfare_per_iteration(self):
        logging.debug('')
        logging.debug('Calculate current efficiency:')
        allocation, objective = self.solve_WDP(self.time_limit, self.elicited_bids, verbose=1)
        self.efficient_allocation_per_iteration[self.mlca_iteration] = allocation
        efficiency = self.calculate_efficiency_of_allocation(allocation=allocation, allocation_scw=objective) # -1 elicited bids after previous iteration
        self.efficiency_per_iteration[self.mlca_iteration] = efficiency
        self.welfare_per_iteration[self.mlca_iteration] = objective 
       # self.exp_avg_rule[self.mlca_iteration+1] = np.average(list(self.welfare_per_iteration.values())) # no prior instances solved to optimality, only reward average

    def set_NN_parameters(self, parameters):
        logging.debug('Set NN parameters')
        self.NN_parameters = OrderedDict(parameters)
    def set_NN_ALLOC_parameters(self, parameters):
        logging.debug('Set NN ALLOC parameters')
        self.NN_Alloc_parameters = OrderedDict(parameters)

    def set_initial_bids(self, initial_bids=None, fitted_scaler=None):
        logging.info('INITIALIZE BIDS')
        logging.info('-----------------------------------------------\n')
        if initial_bids is None: # Uniform sampling
            _elicited, _fitted = util.initial_bids_mlca_unif(SATS_auction_instance=self.SATS_auction_instance,
                                                                                 number_initial_bids=self.Qinit, bidder_names=self.bidder_names,
                                                                                 scaler=self.scaler)
            self.elicited_bids, self.fitted_scaler = deepcopy(_elicited),deepcopy(_fitted)
            # added initial bids to capture initialization
            self.initial_bundles = deepcopy(_elicited)
        else:
            logging.debug('Setting inputed initial bids of dimensions:')
            if not (list(initial_bids.keys()) == self.bidder_names):
                logging.info('Cannot set inputed initial bids-> sample uniformly.') # Uniform sampling
                self.elicited_bids, self.fitted_scaler = util.initial_bids_mlca_unif(SATS_auction_instance=self.SATS_auction_instance,
                                                                                 number_initial_bids=self.Qinit,bidder_names=self.bidder_names,
                                                                                 scaler=self.scaler)
            else:
                for k,v in initial_bids.items():
                    logging.debug(k + ': X=%s, Y=%s',v[0].shape, v[1].shape)
                _elicited = initial_bids # needed format: {Bidder_i:[bundles,values]}
                self.elicited_bids = deepcopy(_elicited)
                # added initial bids to capture initialization
                self.initial_bundles = deepcopy(_elicited)
                self.fitted_scaler = fitted_scaler  # fitted scaler to initial bids
                self.Qinit = [v[0].shape[0] for k, v in initial_bids.items()]


    def set_initial_bids_mlca_extra(self, initial_bids=None, fitted_scaler=None):
        logging.info('INITIALIZE BIDS')
        logging.info('-----------------------------------------------\n')
        if initial_bids is None: # Uniform sampling
            _elicited, _fitted = util.initial_bids_mlca_extra(SATS_auction_instance=self.SATS_auction_instance,
                                                                                 number_initial_bids=self.Qinit, bidder_names=self.bidder_names,
                                                                                 scaler=self.scaler)

            self.elicited_bids, self.fitted_scaler =deepcopy(_elicited),_fitted
            #added initial bids to capture initialization
            self.initial_bundles = deepcopy(_elicited)
        else:
            logging.debug('Setting inputed initial bids of dimensions:')
            if not (list(initial_bids.keys()) == self.bidder_names):
                logging.info('Cannot set inputed initial bids-> sample uniformly.') # Uniform sampling
                _elicited, _fitted = util.initial_bids_mlca_extra(SATS_auction_instance=self.SATS_auction_instance,
                                                                                 number_initial_bids=self.Qinit,bidder_names=self.bidder_names,
                                                                                 scaler=self.scaler)
                self.elicited_bids, self.fitted_scaler = deepcopy(_elicited), _fitted
                # added initial bids to capture initialization
                self.initial_bundles = deepcopy(_elicited)

            else:
                for k,v in initial_bids.items():
                    logging.debug(k + ': X=%s, Y=%s',v[0].shape, v[1].shape)
                _elicited = initial_bids # needed format: {Bidder_i:[bundles,values]}
                self.elicited_bids = deepcopy(_elicited)
                # added initial bids to capture initialization
                self.initial_bundles = deepcopy(_elicited)
                self.fitted_scaler = fitted_scaler # fitted scaler to initial bids
                self.Qinit = [v[0].shape[0] for k,v in initial_bids.items()]


    def set_initial_bids_unif_empty(self, initial_bids=None, fitted_scaler=None):
        logging.info('INITIALIZE BIDS')
        logging.info('-----------------------------------------------\n')
        if initial_bids is None: # Uniform sampling
            _elicited, _fitted = util.initial_bids_mlca_unif_empty(SATS_auction_instance=self.SATS_auction_instance,
                                                                                 number_initial_bids=self.Qinit, bidder_names=self.bidder_names,
                                                                                 scaler=self.scaler)

            self.elicited_bids, self.fitted_scaler =deepcopy(_elicited),_fitted
            #added initial bids to capture initialization
            self.initial_bundles = deepcopy(_elicited)
        else:
            logging.debug('Setting inputed initial bids of dimensions:')
            if not (list(initial_bids.keys()) == self.bidder_names):
                logging.info('Cannot set inputed initial bids-> sample uniformly.') # Uniform sampling
                _elicited, _fitted = util.initial_bids_mlca_unif_empty(SATS_auction_instance=self.SATS_auction_instance,
                                                                                 number_initial_bids=self.Qinit,bidder_names=self.bidder_names,
                                                                                 scaler=self.scaler)
                self.elicited_bids, self.fitted_scaler = deepcopy(_elicited), _fitted
                # added initial bids to capture initialization
                self.initial_bundles = deepcopy(_elicited)

            else:
                for k,v in initial_bids.items():
                    logging.debug(k + ': X=%s, Y=%s',v[0].shape, v[1].shape)
                _elicited = initial_bids # needed format: {Bidder_i:[bundles,values]}
                self.elicited_bids = deepcopy(_elicited)
                # added initial bids to capture initialization
                self.initial_bundles = deepcopy(_elicited)
                self.fitted_scaler = fitted_scaler # fitted scaler to initial bids
                self.Qinit = [v[0].shape[0] for k,v in initial_bids.items()]



    def reset_argmax_allocations(self):
        self.argmax_allocation = OrderedDict(list((key, 
                        OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in self.economies_names.items()))

    def reset_current_query_profile(self):
        self.current_query_profile =  OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))

    def reset_NN_models(self):
        delattr(self, 'NN_models')
        self.NN_models = OrderedDict(list((key, 
            OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in self.economies_names.items()))

    def reset_economy_status(self):
        self.economy_status = OrderedDict(list((key, False) for key, value in self.economies.items()))

    def solve_SATS_auction_instance(self):
        self.SATS_auction_instance_allocation, self.SATS_auction_instance_scw = self.SATS_auction_instance.get_efficient_allocation()

    def sample_marginal_economies(self, active_bidder, number_of_marginals):
        admissible_marginals = [x for x in list((self.economies.keys())) if x not in ['Main Economy', 'Marginal Economy -({})'.format(active_bidder)]]
        return random.sample(admissible_marginals, k=number_of_marginals)

    def update_elicited_bids(self):
        for bidder in self.bidder_names:
            logging.info('UPDATE ELICITED BIDS: S -> R for %s', bidder)
            logging.info('---------------------------------------------')
            # update bundles
            self.elicited_bids[bidder][0] = np.append(self.elicited_bids[bidder][0],self.current_query_profile[bidder], axis=0)
            # update values
            bidder_value_reports = self.value_queries(bidder_id=util.key_to_int(bidder), bundles=self.current_query_profile[bidder])
            self.elicited_bids[bidder][1] = np.append(self.elicited_bids[bidder][1],bidder_value_reports, axis=0) # update values
            logging.info('CHECK Uniqueness of updated elicited bids:')
            check = len(np.unique(self.elicited_bids[bidder][0], axis=0))==len(self.elicited_bids[bidder][0])
            logging.info('UNIQUE\n') if check else logging.debug('NOT UNIQUE\n')
        return(check)

    def update_current_query_profile(self, bidder, bundle_to_add):
        if bundle_to_add.shape != (self.M,):
            logging.debug('No valid bundle dim -> CANNOT ADD BUNDLE')
            return(False)
        if self.current_query_profile[bidder] is None: # If empty can add bundle_to_add for sure
            logging.debug('Current query profile is empty -> ADD BUNDLE')
            self.current_query_profile[bidder] = bundle_to_add.reshape(1,-1)
            return(True)
        else: # If not empty, first check for duplicates, then add
            if self.check_bundle_contained(bundle=bundle_to_add, bidder=bidder):
                return(False)
            else:
                self.current_query_profile[bidder] = np.append(self.current_query_profile[bidder], bundle_to_add.reshape(1, -1), axis=0)
                logging.debug('ADD BUNDLE to current query profile')
                return(True)

    def value_queries(self, bidder_id, bundles):
        raw_values = np.array([self.SATS_auction_instance.calculate_value(bidder_id, bundles[k,:]) for k in range(bundles.shape[0])])
        if self.fitted_scaler is None:
            logging.debug('Return raw value queries')
            return (raw_values)
        else:
            minI = int(round(self.fitted_scaler.data_min_[0]*self.fitted_scaler.scale_[0]))
            maxI = int(round(self.fitted_scaler.data_max_[0]*self.fitted_scaler.scale_[0]))
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug('raw values %s', raw_values)
            logging.debug('Return value queries scaled by: %s to the interval [%s,%s]',round(self.fitted_scaler.scale_[0],8),minI,maxI)
            logging.debug('scaled values %s', self.fitted_scaler.transform(raw_values.reshape(-1,1)).flatten())
            logging.debug('---------------------------------------------')
            return (self.fitted_scaler.transform(raw_values.reshape(-1,1)).flatten())

    def check_bundle_contained(self, bundle, bidder):
        if np.any(np.equal(self.elicited_bids[bidder][0],bundle).all(axis=1)):
            logging.info('Argmax bundle ALREADY ELICITED from {}\n'.format(bidder))
            return(True)
        if self.current_query_profile[bidder] is not None:
            if np.any(np.equal(self.current_query_profile[bidder], bundle).all(axis=1)):
                logging.info('Argmax bundle ALREADY QUERIED IN THIS AUCTION ROUND from {}\n'.format(bidder))
                return(True)
        return(False)

    def next_queries(self, economy_key, active_bidders):
        R_union_S = dict()
        for active_bidder in active_bidders:
            if self.current_query_profile[active_bidder] is not None: # recalc optimization step with bidder specific constraints added
                    Ri_union_Si = np.append(self.elicited_bids[active_bidder][0], self.current_query_profile[active_bidder],axis=0)
            else:
                    Ri_union_Si = self.elicited_bids[active_bidder][0]
            R_union_S[active_bidder] = torch.tensor(Ri_union_Si).cuda()

        if not self.economy_status[economy_key]:  # check if economy already has been calculated prior
            self.estimation_step(economy_key=economy_key)
            self.optimization_step(economy_key=economy_key, R_union_S=R_union_S)

        # if self.check_bundle_contained(bundle=self.argmax_allocation[economy_key][active_bidder][0],
        #                                bidder=active_bidder): # Check if argmax bundle already has been queried in R or S
        #     if self.current_query_profile[active_bidder] is not None: # recalc optimization step with bidder specific constraints added
        #         Ri_union_Si = np.append(self.elicited_bids[active_bidder][0], self.current_query_profile[active_bidder],axis=0)
        #     else:
        #         Ri_union_Si = self.elicited_bids[active_bidder][0]
        #     self.nonunique_count += 1
        #     CTs = OrderedDict()
        #     CTs[active_bidder] = Ri_union_Si
        #     self.find_a_query(economy_key, bidder_specific_constraints=CTs) #we eliminate optimization
        #     self.economy_status[economy_key] = True  # set status of economy to true
        #     return(self.argmax_allocation[economy_key][active_bidder][1])  # return constrained argmax bundle

        # else: # If argmax bundle has NOT already been queried
        # self.unique_count += 1
        self.economy_status[economy_key]=True  # set status of economy to true
        q_s = {active_bidder: self.argmax_allocation[economy_key][active_bidder][0] for active_bidder in active_bidders}
        # return (self.argmax_allocation[economy_key][active_bidder][0])
        return q_s # return regular argmax bundle

    def estimation_step(self, economy_key):
        logging.info('ESTIMATON STEP')
        logging.info('-----------------------------------------------')
        models = OrderedDict()
        for bidder in self.economies_names[economy_key]:
            bids = self.elicited_bids[bidder]
          #  print('Elicited bids have dimensionality of {} (rows) - {} - cols'.format(len(bids), len(bids[0])))
            logging.info(bidder)

            start = time.time()
            nn_model = MLCA_NN(X_train=bids[0], Y_train=bids[1], scaler=self.fitted_scaler)  # instantiate class
            nn_model.initialize_model(model_parameters=self.NN_parameters[bidder])  # initialize model
            tmp = nn_model.fit(epochs=self.NN_parameters[bidder]['epochs'], batch_size=self.NN_parameters[bidder]['batch_size'],  # fit model to data
                               X_valid=None, Y_valid=None)
            end = time.time()
            logging.info('Time for ' + bidder + ': %s sec\n', round(end-start))
            self.losses[economy_key][bidder].append(tmp)
            models[bidder] = nn_model
        self.NN_models[economy_key] = models

    def optimization_step(self, economy_key, R_union_S, bidder_specific_constraints=None):

        DNNs = OrderedDict(list((key, self.NN_models[economy_key][key].model) for key in list(self.NN_models[economy_key].keys())))

        if bidder_specific_constraints is None:
            logging.info('OPTIMIZATION STEP')
        else:
            logging.info('ADDITIONAL BIDDER SPECIFIC OPTIMIZATION STEP for {}'
                         .format(list(bidder_specific_constraints.keys())[0]))
        logging.info('-----------------------------------------------')
        start = time.time()


        X = MLCA_NN_ALLOC(DNNs)
        tmp = X.fit_alloc(self.baseline, model_parameters_alloc=self.NN_Alloc_parameters, R_union_S=R_union_S).detach().to("cpu").numpy()

        end = time.time()
        logging.info('Time for ' + economy_key + ': %s sec\n', round(end - start))
        print(f'Total time for this step for {economy_key} was {(end - start)}')

        if bidder_specific_constraints is None:
            logging.debug('SET ARGMAX ALLOCATION FOR ALL BIDDERS')
            b = 0
            for bidder in self.argmax_allocation[economy_key].keys():
                self.argmax_allocation[economy_key][bidder][0] = tmp[b, :]
                b = b + 1
        #here we call find a query which outputs a query that tries to diminish information on specific bundles
        else:
            logging.debug('SET ARGMAX ALLOCATION ONLY BIDDER SPECIFIC for {}'.format(list(bidder_specific_constraints.keys())[0]))
            self.find_a_query(economy_key, bidder_specific_constraints)

        for key, value in self.argmax_allocation[economy_key].items():
            logging.debug(key + ':  %s | %s', value[0], value[1])

        del X
        del DNNs

    def find_a_query(self, economy_key, bidder_specific_constraints):

        for bidder in bidder_specific_constraints.keys():
            logging.debug('Finding a Query with FREQ {}'.format(bidder))
            self.current_bundle = bidder_specific_constraints[bidder]
            self.summa_row = np.sum(self.current_bundle, axis=0, dtype=float)  # get the sum over rows
            self.summa_columns = np.sum(self.current_bundle, axis=1, dtype=float)  # get the sum over columns
            self.normalize = np.abs(self.summa_row - np.max(self.summa_row)) # first find the max over columns then make the indices with max 0 and the remaining positive
            self.indice_array = np.arange(0, self.M, 1)  # create indice array
            self.new_query = np.zeros(shape=(self.M))


            #this is for picking a bundle size that is not asked
            if np.sum(self.normalize) == 0:
                logging.debug('New query will be found by randomizing bundle size')
                self.new_query[np.random.choice(self.indice_array, size=random.choice(np.setdiff1d(self.indice_array, np.unique(self.summa_columns))),
                                      replace=False)] = 1 #picks a bundle size that is not picked before
                if np.any(np.equal(bidder_specific_constraints[bidder], self.new_query).all(axis=1)):
                    self.find_a_query(economy_key, bidder_specific_constraints)

                self.argmax_allocation[economy_key][bidder][1] = self.new_query

            #this is for picking the least seen item
            else:
                logging.debug('New query will be found by checking frequencies of items')
                self.ttl_vl = np.sum(self.normalize)
                self.prob = (self.normalize+1e-1) / (self.ttl_vl+self.M*1e-1)
                logging.debug('The prob: %s',self.prob)
                self.non_zero = np.nonzero(self.prob)[0].size
                self.rand_pick = np.unique(np.random.choice(np.arange(0, self.M), size=self.non_zero, p=self.prob))
                self.new_query[self.rand_pick] = 1
                logging.debug('The new query: %s', self.new_query)
                if np.any(np.equal(bidder_specific_constraints[bidder], self.new_query).all(axis=1)):
                    self.find_a_query(economy_key, bidder_specific_constraints)

                self.argmax_allocation[economy_key][bidder][1] = self.new_query

    def calculate_mlca_allocation(self, economy='Main Economy'):
        logging.info('Calculate MLCA allocation: %s', economy)
        active_bidders = self.economies_names[economy]
        logging.debug('Active bidders: %s', active_bidders)
        allocation, objective = self.solve_WDP(self.time_limit, elicited_bids=OrderedDict(list((k, self.elicited_bids.get(k, None)) for k in active_bidders)))
        for key,value in allocation.items():
            logging.debug('%s %s', key, value)
        logging.debug('Social Welfare: %s', objective)
        # setting allocations
        if economy == 'Main Economy':
            self.mlca_allocation = allocation
            self.mlca_scw = objective
        if economy in self.mlca_marginal_allocations.keys():
            self.mlca_marginal_allocations[economy] = allocation
            self.mlca_marginal_scws[economy] = objective

    def solve_WDP(self, time_limit, elicited_bids, verbose=0):  # rem.: objective always rescaled to true values
        bidder_names = list(elicited_bids.keys())
        if verbose==1: logging.debug('Solving WDP based on elicited bids for bidder: %s', bidder_names)
        elicited_bundle_value_pairs = [np.concatenate((bids[0], bids[1].reshape(-1, 1)), axis=1) for bidder, bids in elicited_bids.items()]                   # transform self.elicited_bids into format for WDP class
        wdp = WDP(elicited_bundle_value_pairs)
        wdp.initialize_mip(verbose=0)
        wdp.solve_mip(time_limit, verbose)
        #TODO: check solution formater
        objective = wdp.Mip.objective_value
        allocation = util.format_solution_mip_new(Mip=wdp.Mip, elicited_bids=elicited_bundle_value_pairs,
                                                     bidder_names=bidder_names, fitted_scaler=self.fitted_scaler)
        if self.fitted_scaler is not None:
            if verbose==1:
                logging.debug('')
                logging.debug('*SCALING*')
                logging.debug('---------------------------------------------')
                logging.debug('WDP objective scaled: %s:', objective)
                logging.debug('WDP objective value scaled by: 1/%s',round(self.fitted_scaler.scale_[0],8))
            objective = float(self.fitted_scaler.inverse_transform([[objective]]))
            if verbose==1:
                logging.debug('WDP objective orig: %s:', objective)
                logging.debug('---------------------------------------------')
        return(allocation, objective)



    def calculate_efficiency_of_allocation(self, allocation, allocation_scw, verbose=0):
        self.solve_SATS_auction_instance()
        efficiency =  allocation_scw/self.SATS_auction_instance_scw
        if verbose==1:
            logging.debug('Calculating efficiency of input allocation:')
            for key,value in allocation.items():
                logging.debug('%s %s', key, value)
            logging.debug('Social Welfare of allocation: %s', allocation_scw)
            logging.debug('Efficiency of allocation: %s', efficiency)
        return(efficiency)

    def calculate_vcg_payments(self, forced_recalc=False):
        logging.debug('Calculate payments')

        # (i) solve marginal MIPs
        for economy in list(self.economies_names.keys()):
            if not forced_recalc:
                if economy == 'Main Economy' and self.mlca_allocation is None:
                    self.calculate_mlca_allocation()
                elif economy in self.mlca_marginal_allocations.keys() and self.mlca_marginal_allocations[economy] is None:
                    self.calculate_mlca_allocation(economy=economy)
                else:
                    logging.debug('Allocation for %s already calculated', economy)
            else:
                logging.debug('Forced recalculation of %s', economy)
                self.calculate_mlca_allocation(economy=economy) # Recalc economy

        # (ii) calculate VCG terms for this economy
        for bidder in self.bidder_names:
            marginal_economy_bidder = 'Marginal Economy -({})'.format(util.key_to_int(bidder))
            p1 = self.mlca_marginal_scws[marginal_economy_bidder]  # social welfare of the allocation a^(-i) in this economy
            p2 = sum([self.mlca_allocation[i]['value'] for i in self.economies_names[marginal_economy_bidder]])  # social welfare of mlca allocation without bidder i
            self.mlca_payments[bidder] = round(p1-p2,2)
            logging.info('Payment %s: %s - %s  =  %s', bidder, p1, p2, self.mlca_payments[bidder])
        revenue = sum([self.mlca_payments[i] for i in self.bidder_names])
        logging.info('Revenue: {} | {}% of SCW in efficienct allocation\n'.format(revenue, revenue/self.SATS_auction_instance_scw))
# %%
print('MLCA_Economies imported')