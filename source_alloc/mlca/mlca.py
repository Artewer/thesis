# -*- coding: utf-8 -*-

# Libs
from datetime import datetime
import logging
import random
import numpy as np
import time
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

# Own Modules
# from sats.pysats import PySats
from source_alloc.pysats import PySats
from source_alloc.mlca.mlca_economies import MLCA_Economies
from source_alloc.util import key_to_int, timediff_d_h_m_s


#%% MLCA MECHANISM SINGLE RUN
def mlca_mechanism(SATS_domain_name=None, SATS_auction_instance_seed=None, Qinit=None, Qmax=None, Qround=None, NN_parameters=None, MIP_parameters=None, scaler=None,
                   init_bids_and_fitted_scaler=[None,None], return_allocation=False, return_payments=False, calc_efficiency_per_iteration=False, configdict=None):

    start = datetime.now()

    isLegacy = False

    #can also specify input as witha dict of an configuration
    if configdict is not None:
        SATS_domain_name = configdict['SATS_domain_name']
        SATS_auction_instance_seed = configdict['SATS_auction_instance_seed']
        Qinit = configdict['Qinit']
        Qmax = configdict['Qmax']
        Qround = configdict['Qround']
        NN_parameters = configdict['NN_parameters']
        NN_ALLOC_parameters = configdict['NN_Alloc_parameters']
        scaler = configdict['scaler']
        init_bids_and_fitted_scaler = configdict['init_bids_and_fitted_scaler']
        return_allocation = configdict['return_allocation']
        return_payments = configdict['return_payments']
        calc_efficiency_per_iteration = configdict['calc_efficiency_per_iteration']
        # starter = configdict['Starter']



    logging.warning('START MLCA:')
    logging.warning('-----------------------------------------------')
    OUTPUT={}
    logging.warning('Model: %s',SATS_domain_name)
    logging.warning('Seed SATS Instance: %s',SATS_auction_instance_seed)
    logging.warning('Qinit: %s',Qinit)
    logging.warning('Qmax: %s',Qmax)
    logging.warning('Qround: %s\n',Qround)


    # Average welfare over 5 random SATS instances, used as initial baseline
    sats_scw = {}
    init_scw = 0
    time_per_round = []
    
    '''
    for i in range(10):
        current_seed = random.randint(0,650)
        random.seed(current_seed)
        init_instance = PySats.getInstance().create_gsvm(seed=current_seed)  # create SATS auction instance
        sats_alloc, sats_scw[i] = init_instance.get_efficient_allocation()
        print(f'Instance {current_seed}, SW equals {sats_scw[i]}')
       
    init_scw = np.mean(list(sats_scw.values()))
    print(f'Average SW equals {init_scw}')
    '''

    random.seed(SATS_auction_instance_seed)  # (*) truly uniform bids: global seed

    # Instantiate Economies
    logging.debug('Instantiate SATS Instance')
    
    # if SATS_domain_name == 'LSVM':
    #     SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_auction_instance_seed,
    #                                                              isLegacyLSVM=isLegacy)  # create SATS auction instance
    #     logging.warning('####### ATTENTION #######')
    #     logging.warning('isLegacyLSVM: %s', SATS_auction_instance.isLegacy)
    #     logging.warning('#########################\n')
    # if SATS_domain_name == 'GSVM':
    #     SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_auction_instance_seed,
    #                                                              isLegacyGSVM=isLegacy)  # create SATS auction instance
    #     logging.warning('####### ATTENTION #######')
    #     logging.warning('isLegacyGSVM: %s', SATS_auction_instance.isLegacy)
    #     logging.warning('#########################\n')
    # if SATS_domain_name == 'MRVM':
    #     SATS_auction_instance = PySats.getInstance().create_mrvm(
    #         seed=SATS_auction_instance_seed)  # create SATS auction instance
    # if SATS_domain_name == 'SRVM':
    #     SATS_auction_instance = PySats.getInstance().create_srvm(
    #         seed=SATS_auction_instance_seed)  # create SATS auction instance

    if SATS_domain_name == 'LSVM':
        SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_auction_instance_seed)  # create SATS auction instance
    if SATS_domain_name == 'GSVM':
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_auction_instance_seed)  # create SATS auction instance
    if SATS_domain_name == 'MRVM':
        SATS_auction_instance = PySats.getInstance().create_mrvm(seed=SATS_auction_instance_seed)  # create SATS auction instance
    
    
    E = MLCA_Economies(SATS_auction_instance=SATS_auction_instance, SATS_auction_instance_seed=SATS_auction_instance_seed,
                       Qinit=Qinit, Qmax=Qmax, Qround=Qround, scaler=scaler)  # create economy instance

    E.set_NN_parameters(parameters=NN_parameters)   # set NN parameters
    E.set_NN_ALLOC_parameters(parameters=NN_ALLOC_parameters)   # set MIP parameters
    
    E.exp_avg_rule[0] = init_scw  # in case no prior optimal instances used, value set to 0

    # Set initial bids | Line 1-3
    # init_bids, init_fitted_scaler = init_bids_and_fitted_scaler
    # if init_bids is not None:
    #     if starter =='unif_empty':
    #         E.set_initial_bids_unif_empty(initial_bids=init_bids, fitted_scaler=init_fitted_scaler) # (*) use self defined inital bids | Line 1
    #     elif starter == 'mlca_extra':
    #         E.set_initial_bids_mlca_extra(initial_bids=init_bids, fitted_scaler=init_fitted_scaler) # (*) use self defined inital bids | Line 1
    # else:
    #     if starter == 'unif_empty':
    #         E.set_initial_bids_unif_empty()
    #     elif starter == 'mlca_extra':
    #         E.set_initial_bids_mlca_extra()
        # Set initial bids | Line 1-3

    init_bids, init_fitted_scaler = init_bids_and_fitted_scaler
    if init_bids is not None:
        E.set_initial_bids(initial_bids=init_bids,
                           fitted_scaler=init_fitted_scaler)  # (*) use self defined inital bids | Line 1
    else:
        E.set_initial_bids()  # (*) create inital bids B^0, uniformly sampling at random from bundle space | Line 2
        E.calculate_mlca_allocation()

    # Calculate efficient allocation given current elicited bids
    if calc_efficiency_per_iteration: E.calculate_efficiency_welfare_per_iteration() 

    # Global while loop: check if for all bidders one addtitional auction round is feasible | Line 4
    Rmax = max(E.get_number_of_elicited_bids().values())
    CHECK = Rmax <= (E.Qmax-E.Qround)
    
    while CHECK:

        round_start = time.time()

        # Increment iteration
        E.mlca_iteration+=1
        # log info
        E.get_info()

        # Check Current Query Profile: | Line 5
        logging.debug('Current query profile S=(S_1,...,S_n):')
        for k,v in E.current_query_profile.items():
            logging.debug(k+':  %s',v)

    
        # Marginal Economies: | Line 6-12
        logging.info('MARGINAL ECONOMIES FOR ALL BIDDERS')
        logging.info('-----------------------------------------------\n')

        sampled_marginal_economies_bidders = dict()

        for bidder in E.bidder_names:
            logging.info(bidder)
            logging.info('-----------------------------------------------')
            logging.debug('Sampling marginals for %s', bidder)
            sampled_marginal_economies = E.sample_marginal_economies(active_bidder=key_to_int(bidder), number_of_marginals=E.Qround-1)
            sampled_marginal_economies.sort()
            sampled_marginal_economies_bidders[bidder] = sampled_marginal_economies
            logging.debug('Calculate next queries for the following sampled marginals:')
            for marginal_economy in sampled_marginal_economies:
                logging.debug(marginal_economy)
        
        sampled_marginal_economies_all = dict()
        for economy in E.economies.keys():
            for bidder in E.bidder_names:
                if economy in sampled_marginal_economies_bidders[bidder]:
                    if economy not in sampled_marginal_economies_all:
                        sampled_marginal_economies_all[economy] = [bidder]  
                    else:
                        sampled_marginal_economies_all[economy].append(bidder)   
        

        for marginal_economy, active_bidders in sampled_marginal_economies_all.items():   
           # for marginal_economy in sampled_marginal_economies:
           #     logging.debug('')
           #     logging.info(bidder + ' | '+ marginal_economy)
           #     logging.info('-----------------------------------------------')
            logging.info('Status of Economy: %s\n', E.economy_status[marginal_economy])
            #    q_i = E.next_queries(economy_key=marginal_economy, active_bidders= [bidder])
            #    q_i = E.next_queries(economy_key=marginal_economy, active_bidders=active_bidders)
            q_i_s = E.next_queries(economy_key=marginal_economy, active_bidders=active_bidders)
            for bidder in active_bidders:
                q_i = q_i_s[bidder]
                E.update_current_query_profile(bidder=bidder, bundle_to_add=q_i)
                # logging.debug('')
                # logging.debug('Current query profile for %s:', bidder)
                # for bundle in range(E.current_query_profile[bidder].shape[0]):
                #     logging.debug(E.current_query_profile[bidder][k,:])
                # logging.debug('')

                # q_i = q_i[bidder]
                # E.update_current_query_profile(bidder=bidder, bundle_to_add=q_i)
                # logging.debug('')
                # logging.debug('Current query profile for %s:', bidder)
                # for k in range(E.current_query_profile[bidder].shape[0]):
                #     logging.debug(E.current_query_profile[bidder][k,:])
                # logging.debug('')
        
        
        # Main Economy: | Line 13-14
        logging.info('MAIN ECONOMY FOR ALL BIDDERS')
        logging.info('-----------------------------------------------\n')
       # for bidder in E.bidder_names:
       #     logging.info(bidder)
        logging.info('-----------------------------------------------')
        economy_key = 'Main Economy'
        logging.debug(economy_key)
        logging.debug('-----------------------------------------------')
        logging.debug('Status of Economy: %s', E.economy_status[economy_key])
        print(f'Status of Economy: {E.economy_status[economy_key]}')
        q_i_s = E.next_queries(economy_key=economy_key, active_bidders=E.bidder_names)
        for bidder in E.bidder_names:
            q_i = q_i_s[bidder]
            E.update_current_query_profile(bidder=bidder, bundle_to_add=q_i)
            # logging.debug('')
            # logging.debug('Current query profile for %s:', bidder)
            # for bundle in range(E.current_query_profile[bidder].shape[0]):
            #     logging.debug(E.current_query_profile[bidder][k,:])
            # logging.debug('')

        # Update Elicited Bids With Current Query Profile and check uniqueness | Line 15-16
        if not E.update_elicited_bids(): return('UNIQUENESS CHECK FAILED see logfile')


        # Reset Attributes | Line 18
        logging.info('RESET: Auction Round Query Profile S=(S_1,...,S_n)')
        E.reset_current_query_profile()
        logging.info('RESET: Status of Economies')
        E.reset_economy_status()
        logging.info('RESET: NN Models')
        E.reset_NN_models()
        logging.info('RESET: Argmax Allocation\n')
        E.reset_argmax_allocations()
        # clear_session()  # clear keras session

        # Calculate efficient allocation given current elicited bids
        # if calc_efficiency_per_iteration: E.calculate_efficiency_per_iteration()
        
        if calc_efficiency_per_iteration: E.calculate_efficiency_welfare_per_iteration()

        # Compute baseline by letting the MIP run on elicited bundles for a few seconds
        # E.run_mip_on_elicited(1) 
        # print(f'MIP objective is {E.baseline}')

        # Update while condition
        Rmax = max(E.get_number_of_elicited_bids().values())
        CHECK = Rmax <= (E.Qmax-E.Qround)

        round_end = time.time()
        time_per_round.append((round_end- round_start))
        print(f'For iteration {E.mlca_iteration}, time is {(round_end-round_start)}')

    

    # allocation & payments # | Line 20
    if return_allocation:
        logging.info('ALLOCATION')
        logging.info('---------------------------------------------')
        E.calculate_mlca_allocation()
        E.mlca_allocation_efficiency = E.calculate_efficiency_of_allocation(E.mlca_allocation, E.mlca_scw, verbose=1)
    # Payments - marginal economiess
    if return_payments: # | Line 21
        logging.info('')
        logging.info('PAYMENTS')
        logging.info('---------------------------------------------')
        E.calculate_vcg_payments()
    

    end = datetime.now()


    total_time_elapsed = '{}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(end-start))
    E.get_info(final_summary=True)
    logging.warning('EFFICIENCY: {} %'.format(round(E.mlca_allocation_efficiency,4)*100))
    logging.warning('TOTAL TIME ELAPSED: {}'.format(total_time_elapsed))
    logging.warning('MLCA FINISHED')

    # Set OUTPUT
    OUTPUT['SEED TRUE VALUES'] = {'Seed': E.SATS_auction_instance_seed, 'SCW': E.SATS_auction_instance_scw,
                                  'Allocation': E.SATS_auction_instance_allocation}
    OUTPUT['MLCA Efficiency'] = E.mlca_allocation_efficiency
    OUTPUT['MLCA Allocation'] = E.mlca_allocation
    OUTPUT['MLCA Payments'] = E.mlca_payments
    OUTPUT['Statistics']={'Total Time Elapsed':total_time_elapsed, 'Time Per Round': time_per_round, 'Elapsed Times of MIPs': E.elapsed_time_mip,
                          'NN Losses': E.losses, 'Efficiency per Iteration': E.efficiency_per_iteration,
                          'Efficient allocation per Iteration': E.efficient_allocation_per_iteration}
    OUTPUT['Initial_Bundles']=E.initial_bundles
    OUTPUT['Elicited Bids'] = E.elicited_bids

    return(OUTPUT)
#%#
print('MLCA function imported')