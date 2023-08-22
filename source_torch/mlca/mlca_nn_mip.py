# -*- coding: utf-8 -*-

""

# Libs
import pandas as pd
import numpy as np
import logging
from collections import OrderedDict
import re
# CPLEX: Here, DOcplex is used for solving the deep neural network-based Winner Determination Problem.
import docplex.mp.model as cpx
# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html

import pdb


# %% Neural Net Optimization Class


class MLCA_NNMIP:

    def __init__(self, models, L=None):
        #M = models[list(models.keys())[0]].model[0].weight.data.cpu().T.numpy().shape[0] # number of items in the value model = dimension of input layer
        #print(M)
        model_name = list(models.keys())[0]
        models[model_name].model.cuda()
        self.M = models[model_name].model[0].weight.T.data.shape[0]
        print(self.M)

        self.Models = models  # dict of keras models
        #put all models on gpu
        # sorted list of bidders
        self.sorted_bidders = list(self.Models.keys())
        self.sorted_bidders.sort()
        self.N = len(models)  # number of bidders
        self.Mip = cpx.Model(name="NeuralNetworksMixedIntegerProgram")  # docplex instance
        self.z = {}  # MIP variable: see paper
        self.s = {}  # MIP variable: see paper
        self.y = {}  # MIP variable: see paper
        self.x_star = np.ones((self.N, self.M))*(-1)  # optimal allocation (-1=not yet solved)
        self.L = L  # global big-M variable: see paper
        self.soltime = None  # timing
        self.z_help = {}  # helper variable for bound tightening
        self.s_help = {}  # helper variable for bound tightening
        self.y_help = {}  # helper variable for bound tightening

        # upper bounds for MIP variables z,s. Lower bounds are 0 in our setting.
        self.upper_bounds_z = OrderedDict(list((bidder_name, [np.array([self.L]*layer_shape).reshape(-1, 1) for 
                                                layer_shape in self._get_model_layer_shapes(bidder_name, layer_type=['dense', 'input']) ]) for bidder_name in self.sorted_bidders))
        self.upper_bounds_s = OrderedDict(list((bidder_name, [np.array([self.L]*layer_shape).reshape(-1, 1) for 
                                                layer_shape in self._get_model_layer_shapes(bidder_name, layer_type=['dense', 'input']) ]) for bidder_name in self.sorted_bidders))

    def print_optimal_allocation(self):
        D = pd.DataFrame(self.x_star)
        D.columns = ['Item_{}'.format(j) for j in range(1, self.M+1)]
        D.loc['Sum'] = D.sum(axis=0)
        print(D)

    def solve_mip(self, log_output=False, time_limit=None, mip_relative_gap=None, integrality_tol=None, mip_start=None):
        # add a warm start
        if mip_start is not None:
            self.Mip
            self.Mip.add_mip_start(mip_start)
            print('Adding warm start')
        # set time limit
        if time_limit is not None:
            self.Mip.set_time_limit(time_limit)
            print(f'Time limit is {self.Mip.get_time_limit()}')
        # set mip relative gap
        if mip_relative_gap is not None:
            self.Mip.parameters.mip.tolerances.mipgap = mip_relative_gap
        # set mip integrality tolerance
        if integrality_tol is not None:
            self.Mip.parameters.mip.tolerances.integrality.set(integrality_tol)
        logging.debug('Mip time Limit of %s', self.Mip.get_time_limit())
        logging.debug('Mip relative gap %s', self.Mip.parameters.mip.tolerances.mipgap.get())
        logging.debug('Mip integrality tol %s', self.Mip.parameters.mip.tolerances.integrality.get())
        # solve MIP
        Sol = self.Mip.solve(log_output=log_output)
        # get solution details
        try:
            self.soltime = Sol.solve_details._time
        except Exception:
            self.soltime= None
        self.log_solve_details(self.Mip)
        # set the optimal allocation
        for i in range(0, self.N):
            for j in range(0, self.M):
                self.x_star[i, j] = self.z[(i, 0, j)].solution_value

        return(Sol)

    def log_solve_details(self, solved_mip):
        details = solved_mip.get_solve_details()
        logging.info('-----------------------------------------------')
        logging.info('Status  : %s', details.status)
        logging.info('Time    : %s sec',round(details.time))
        print(f'Time for MIP run: {round(details.time)} sec')
        logging.info('Problem : %s',details.problem_type)
        logging.info('Rel. Gap: {} %'.format(round(details.mip_relative_gap,5)))
        logging.debug('N. Iter : %s',details.nb_iterations)
        logging.debug('Hit Lim.: %s',details.has_hit_limit())
        # logging.debug('Objective Value: %s\n', solved_mip.objective_value)
        print(f'Found objective {solved_mip.objective_value}')

    def summary(self):
        print('################################ OBJECTIVE ################################')
        print(self.Mip.get_objective_expr(), '\n')
        try:
            print('Objective Value: ', self.Mip.objective_value, '\n')
        except Exception:
            print("Objective Value: Not yet solved!\n")
        print('############################# SOLVE STATUS ################################')
        print(self.Mip.get_solve_details())
        print(self.Mip.get_statistics(), '\n')
        try:
            print(self.Mip.get_solve_status(), '\n')
        except AttributeError:
            print("Not yet solved!\n")
        print('########################### OPT ALLOCATION ##############################')
        self.print_optimal_allocation()
        return(' ')

    def print_mip_constraints(self):
        print('############################### CONSTRAINTS ###############################')
        k = 0
        for m in range(0, self.Mip.number_of_constraints):
            if self.Mip.get_constraint_by_index(m) is not None:
                print('({}):   '.format(k), self.Mip.get_constraint_by_index(m))
                k = k+1
            if self.Mip.get_indicator_by_index(m) is not None:
                print('({}):   '.format(k), self.Mip.get_indicator_by_index(m))
                k = k+1
        print('\n')

    # def _get_model_weights(self, key):
    #     Wb = self.Models[key].get_weights()
    #     return(Wb)

    # def _get_model_layers(self, key, layer_type=None):
    #     Layers = self.Models[key].layers
    #     if layer_type is not None:
    #         tmp = [layer.get_config()['name'] for layer in Layers]
    #         Layers = [Layers[i] for i in [tmp.index(s) for s in tmp if any([x in s for x in layer_type])]]
    #     return(Layers)

    def _get_model_weights(self, key): # torch
        nnmodel = self.Models[key].model
        weights = []
        for params in nnmodel.parameters():
            #weights.append(params.detach().T)
            weights.append(params.detach().cpu().numpy().T)        
        return weights

    def _get_model_layer_shapes(self, key, layer_type=None):
        ''' return layer output shapes instead, 
            if 'input' is given as desired layer type, insert input dim at the beginning.
            assumes torch model '''
        # nnmodel = self.Models[key]
        nnmodel = self.Models[key].model
        Layer_shapes = []
        for i, (name, param) in enumerate(nnmodel.named_parameters()):
            if (i==0) and ('input' in layer_type): 
                Layer_shapes.append(param.shape[1])
            if any([x in name for x in layer_type]) and ('bias' in name):
                Layer_shapes.append(param.shape[0])

        return Layer_shapes


    def _clean_weights(self, Wb):
        for v in range(0, len(Wb)-2, 2):
            Wb[v][abs(Wb[v]) <= 1e-8] = 0
            Wb[v+1][abs(Wb[v+1]) <= 1e-8] = 0
            zero_rows = np.where(np.logical_and((Wb[v] == 0).all(axis=0), Wb[v+1] == 0))[0]
            if len(zero_rows) > 0:
                logging.debug('Clean Weights (rows) %s', zero_rows)
                Wb[v] = np.delete(Wb[v], zero_rows, axis=1)
                Wb[v+1] = np.delete(Wb[v+1], zero_rows)
                Wb[v+2] = np.delete(Wb[v+2], zero_rows, axis=0)
        return(Wb)

    def _add_matrix_constraints(self, i, verbose=False):
        layer = 1
        key = self.sorted_bidders[i]
        Wb = self._clean_weights(self._get_model_weights(key))
        # Wb = self._get_model_weights(key)  # old without weights cleaning
        for v in range(0, len(Wb), 2):  # loop over layers
            if verbose is True:
                logging.debug('\nLayer: %s', layer)
            W = Wb[v].transpose()
            if verbose is True:
                logging.debug('W: %s', W.shape)
            b = Wb[v + 1]
            if verbose is True:
                logging.debug('b: %s', b.shape)
            R, J = W.shape
            # decision variables
            if v == 0:
                self.z.update({(i, 0, j): self.Mip.binary_var(name="x({})_{}".format(i, j)) for j in range(0, J)})  # binary variables for allocation

            # pdb.set_trace()
            self.z.update({(i, layer, r): self.Mip.continuous_var(lb=0, name="z({},{})_{}".format(i, layer, r)) for r in range(0, R)})  # output value variables after activation
            self.s.update({(i, layer, r): self.Mip.continuous_var(lb=0, name="s({},{})_{}".format(i, layer, r)) for r in range(0, R) if (self.upper_bounds_z[key][layer][r][0] != 0 and self.upper_bounds_s[key][layer][r][0] != 0)})  # slack variables
            self.y.update({(i, layer, r): self.Mip.binary_var(name="y({},{})_{}".format(i, layer, r)) for r in range(0, R) if (self.upper_bounds_z[key][layer][r][0] != 0 and self.upper_bounds_s[key][layer][r][0] != 0)})  # binary variables for activation function
            # add constraints
            for r in range(0, R):
                if verbose is True:
                    logging.debug('Row: %s', r)
                    logging.debug('W[r,]: %s', W[r, :])
                    logging.debug('b[r]: %s', b[r])
                    logging.debug('upper z-bound: {}, {}, {}, {}'.format(key, layer, r, self.upper_bounds_z[key][layer][r]))
                    logging.debug('upper s-bound: {}, {}, {}, {}'.format(key, layer, r, self.upper_bounds_s[key][layer][r]))
                if self.upper_bounds_z[key][layer][r][0] == 0:
                    if verbose is True:
                        logging.debug('upper z-bound: {}, {}, {} is equal to zero => add z==0 constraints'.format(key, layer, r))
                    self.Mip.add_constraint(ct=self.z[(i, layer, r)] == 0)
                elif self.upper_bounds_s[key][layer][r][0] == 0:
                    if verbose is True:
                        logging.debug('upper s-bound: {}, {}, {} is equal to zero => add z==Wz_pre + b constraints'.format(key, layer, r))
                    self.Mip.add_constraint(ct=(self.Mip.sum(W[r, j]*self.z[(i, layer-1, j)] for j in range(0, J)) + b[r] == self.z[(i, layer, r)]))
                else:
                    self.Mip.add_constraint(ct=(self.Mip.sum(W[r, j]*self.z[(i, layer-1, j)] for j in range(0, J)) + b[r] == self.z[(i, layer, r)] - self.s[(i, layer, r)]),
                                            ctname="AffineCT_Bidder{}_Layer{}_Row{}".format(i, layer, r))
                    # indicator constraints
                    self.Mip.add_constraint(ct=self.z[(i, layer, r)] <= self.y[(i, layer, r)]*self.upper_bounds_z[key][layer][r][0], ctname="BinaryCT_Bidder{}_Layer{}_Row{}_Z".format(i, layer, r))
                    self.Mip.add_constraint(ct=self.s[(i, layer, r)] <= (1-self.y[(i, layer, r)])*self.upper_bounds_s[key][layer][r][0], ctname="BinaryCT_Bidder{}_Layer{}_Row{}_S".format(i, layer, r))
                if verbose is True:
                    for m in range(0, self.Mip.number_of_constraints):
                        if self.Mip.get_constraint_by_index(m) is not None:
                            logging.debug(self.Mip.get_constraint_by_index(m))
                        if self.Mip.get_indicator_by_index(m) is not None:
                            logging.debug(self.Mip.get_indicator_by_index(m))
            layer = layer + 1

    def initialize_mip(self, verbose=False, bidder_specific_constraints=None):
        # pay attention here order is important, thus first sort the keys of bidders!
        logging.debug('Sorted active bidders in MIP: %s', self.sorted_bidders)
        # linear matrix constraints: Wz^(i-1)+b = z^(i)-s^(i)
        for i in range(0, self.N):
            self._add_matrix_constraints(i, verbose=verbose)
        # allocation constraints for x^i's
        for j in range(0, self.M):
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, 0, j)] for i in range(0, self.N)) <= 1), ctname="FeasabilityCT_x({})".format(j))
        # add bidder specific constraints
        if bidder_specific_constraints is not None:
            self._add_bidder_specific_constraints(bidder_specific_constraints)
        # add objective: sum of 1dim outputs of neural network per bidder z[(i,K_i,0)]
        objective = self.Mip.sum(self.z[(i, (len(self._get_model_layer_shapes(self.sorted_bidders[i], layer_type=['dense'])) ), 0)] for i in range(0, self.N))
        self.Mip.maximize(objective)
        logging.debug('Mip initialized')

    #TODO: implmement Gianlucas integer cut
    def _add_bidder_specific_constraints(self, bidder_specific_constraints):
        for bidder_key, bundles in bidder_specific_constraints.items():
                bidder_id = np.where([x==bidder_key for x in self.sorted_bidders])[0][0]
                count=0
                logging.debug('Adding bidder specific constraints')
                for bundle in bundles:
                    #logging.debug(bundle)
                    self.Mip.add_constraint(ct=(self.Mip.sum((self.z[(bidder_id, 0, j)]==bundle[j]) for j in range(0, self.M))<=(self.M-1)),
                                                ctname="BidderSpecificCT_Bidder{}_No{}".format(bidder_id,count))
                    count= count+1

    def get_bidder_key_position(self, bidder_key):
        return np.where([x==bidder_key for x in self.sorted_bidders])[0][0]

    def reset_mip(self):
        self.Mip = cpx.Model(name="MIP")

    def tighten_bounds_IA(self, upper_bound_input, verbose=False):
        for bidder in self.sorted_bidders:
            logging.debug('Tighten bounds with IA for %s', bidder)
            Wb_total = self._clean_weights(self._get_model_weights(bidder))
            k = 0
            for j in range(len(self._get_model_layer_shapes(bidder, layer_type=['dense', 'input']))):  # loop over layers including input layer
                if j == 0:
                    self.upper_bounds_z[bidder][j] = np.array(upper_bound_input).reshape(-1, 1)
                    self.upper_bounds_s[bidder][j] = np.array(upper_bound_input).reshape(-1, 1)
                else:
                    W_plus = np.maximum(Wb_total[k].transpose(), 0)
                    W_minus = np.minimum(Wb_total[k].transpose(), 0)
                    self.upper_bounds_z[bidder][j] = np.ceil(np.maximum(W_plus @ self.upper_bounds_z[bidder][j-1] + Wb_total[k+1].reshape(-1, 1), 0)).astype(int)   # upper bound for z
                    self.upper_bounds_s[bidder][j] = np.ceil(np.maximum(-(W_minus @ self.upper_bounds_z[bidder][j-1] + Wb_total[k+1].reshape(-1, 1)), 0)).astype(int)  # upper bound  for s
                    k = k+2
        if verbose is True:
            logging.debug('Upper Bounds z:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
                logging.debug(v)
            logging.debug('\nUpper Bounds s:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_s.items()}).fillna('-').items():
                logging.debug(v)

    def tighten_bounds_LP(self, upper_bound_input, verbose=False):
        for bidder in self.sorted_bidders:
            logging.debug('Tighten bounds with LPs for %s', bidder)
            i = int(re.findall(r'\d+', bidder)[0])
            Wb_total = self._clean_weights(self._get_model_weights(bidder))
            for layer in range(len(self._get_model_layer_shapes(bidder, layer_type=['dense', 'input']))):  # loop over layers including input layer
                if layer == 0:  # input layer bounds given
                    self.upper_bounds_z[bidder][layer] = np.array(upper_bound_input).reshape(-1, 1)
                    self.upper_bounds_s[bidder][layer] = np.array(upper_bound_input).reshape(-1, 1)
                elif layer == 1:   # first hidden layer no LP needed, can be done like IA
                    W_plus = np.maximum(Wb_total[0].transpose(), 0)
                    W_minus = np.minimum(Wb_total[0].transpose(), 0)
                    self.upper_bounds_z[bidder][layer] = np.ceil(np.maximum(W_plus.sum(axis=1).reshape(-1, 1) + Wb_total[1].reshape(-1, 1), 0)).astype(int)   # upper bound for z
                    self.upper_bounds_s[bidder][layer] = np.ceil(np.maximum(-(W_minus.sum(axis=1).reshape(-1, 1) + Wb_total[1].reshape(-1, 1)), 0)).astype(int)  # upper bound  for s
                else:
                    for k in range(0, len(self.upper_bounds_z[bidder][layer])):
                        if (self.upper_bounds_z[bidder][layer][k][0] == 0 and self.upper_bounds_s[bidder][layer][k][0] == 0):
                            continue
                        helper_Mip = cpx.Model(name="LPBounds")
                        pre_layer = 1
                        for v in range(0, 2*(layer-1), 2):  # loop over prelayers before layer
                            W = Wb_total[v].transpose()
                            b = Wb_total[v + 1]
                            ROWS, COLUMNS = W.shape
                            # initialize decision variables
                            if v == 0:
                                self.z_help.update({(i, 0, j): helper_Mip.binary_var(name="x({})_{}".format(i, j)) for j in range(0, COLUMNS)})  # binary variables for allocation
                            self.z_help.update({(i, pre_layer, r): helper_Mip.continuous_var(lb=0, name="z({},{})_{}".format(i, pre_layer, r)) for r in range(0, ROWS)})  # output value variables after activation
                            self.s_help.update({(i, pre_layer, r): helper_Mip.continuous_var(lb=0, name="s({},{})_{}".format(i, pre_layer, r)) for r in range(0, ROWS) if (self.upper_bounds_z[bidder][pre_layer][r][0] != 0 and self.upper_bounds_s[bidder][pre_layer][r][0] != 0)})  # slack variables
                            self.y_help.update({(i, pre_layer, r): helper_Mip.continuous_var(lb=0, ub=1, name="y({},{})_{}".format(i, pre_layer, r)) for r in range(0, ROWS) if (self.upper_bounds_z[bidder][pre_layer][r][0] != 0 and self.upper_bounds_s[bidder][pre_layer][r][0] != 0)})  # relaxed binary variables for activation function
                            # add constraints
                            for r in range(0, ROWS):
                                if self.upper_bounds_z[bidder][pre_layer][r][0] == 0:
                                    if verbose is True:
                                        logging.debug('upper z-bound: {}, {}, {} is equal to zero => add z==0 constraints'.format(bidder, pre_layer, r))
                                    helper_Mip.add_constraint(ct=self.z_help[(i, pre_layer, r)] == 0)
                                elif self.upper_bounds_s[bidder][pre_layer][r][0] == 0:
                                    if verbose is True:
                                        logging.debug('upper s-bound: {}, {}, {} is equal to zero => add z==Wz_pre + b constraints'.format(bidder, pre_layer, r))
                                    helper_Mip.add_constraint(ct=(helper_Mip.sum(W[r, j]*self.z_help[(i, pre_layer-1, j)] for j in range(0, COLUMNS)) + b[r] == self.z_help[(i, pre_layer, r)]))
                                else:
                                    helper_Mip.add_constraint(ct=(helper_Mip.sum(W[r, j]*self.z_help[(i, pre_layer-1, j)] for j in range(0, COLUMNS)) + b[r] == self.z_help[(i, pre_layer, r)] - self.s_help[(i, pre_layer, r)]),
                                                              ctname="AffineCT_Bidder{}_Layer{}_Row{}".format(i, pre_layer, r))
                                    # relaxed indicator constraints
                                    helper_Mip.add_constraint(ct=self.z_help[(i, pre_layer, r)] <= self.y_help[(i, pre_layer, r)]*self.upper_bounds_z[bidder][pre_layer][r][0], ctname="RelaxedBinaryCT_Bidder{}_Layer{}_Row{}_Z".format(i, pre_layer, r))
                                    helper_Mip.add_constraint(ct=self.s_help[(i, pre_layer, r)] <= (1-self.y_help[(i, pre_layer, r)])*self.upper_bounds_s[bidder][pre_layer][r][0], ctname="RelaxedBinaryCT_Bidder{}_Layer{}_Row{}_S".format(i, pre_layer, r))
                            pre_layer = pre_layer + 1

                        # final extra row constraint
                        W = Wb_total[2*(layer-1)].transpose()
                        b = Wb_total[2*(layer-1) + 1]
                        self.z_help.update({(i, layer, k): helper_Mip.continuous_var(lb=0, name="z({},{})_{}".format(i, layer, k))})
                        self.s_help.update({(i, layer, k): helper_Mip.continuous_var(lb=0, name="s({},{})_{}".format(i, layer, k))})
                        if self.upper_bounds_z[bidder][layer][k][0] == 0:
                            helper_Mip.add_constraint(ct=(helper_Mip.sum(W[k, j]*self.z_help[(i, layer-1, j)] for j in range(0, W.shape[1])) + b[k] == -self.s_help[(i, layer, k)]))
                        elif self.upper_bounds_s[bidder][layer][k][0] == 0:
                            helper_Mip.add_constraint(ct=(helper_Mip.sum(W[k, j]*self.z_help[(i, layer-1, j)] for j in range(0, W.shape[1])) + b[k] == self.z_help[(i, layer, k)]))
                        else:
                            self.y_help.update({(i, layer, k): helper_Mip.continuous_var(lb=0, ub=1, name="y({},{})_{}".format(i, layer, k))})  # relaxed binary variable for activation function for final row constraint
                            helper_Mip.add_constraint(ct=(helper_Mip.sum(W[k, j]*self.z_help[(i, layer-1, j)] for j in range(0, W.shape[1])) + b[k] == self.z_help[(i, layer, k)] - self.s_help[(i, layer, k)]),
                                                      ctname="FinalAffineCT_Bidder{}_Layer{}_Row{}".format(i, layer, k))
                            # final relaxed indicator constraints
                            helper_Mip.add_constraint(ct=self.z_help[(i, layer, k)] <= self.y_help[(i, layer, k)]*self.upper_bounds_z[bidder][layer][k][0], ctname="FinalRelaxedBinaryCT_Bidder{}_Layer{}_Row{}_Z".format(i, layer, k))
                            helper_Mip.add_constraint(ct=self.s_help[(i, layer, k)] <= (1-self.y_help[(i, layer, k)])*self.upper_bounds_s[bidder][layer][k][0], ctname="FinalRelaxedBinaryCT_Bidder{}_Layer{}_Row{}_S".format(i, layer, k))
                        # add objective for z bound only if current bounds larger than zero
                        #helper_Mip.parameters.mip.tolerances.integrality.set(1e-8)
                        if self.upper_bounds_z[bidder][layer][k][0] != 0:
                            helper_Mip.maximize(self.z_help[(i, layer, k)])
                            helper_Mip.solve()
                            self.upper_bounds_z[bidder][layer][k][0] = np.ceil(self.z_help[(i, layer, k)].solution_value).astype(int)
                        if self.upper_bounds_s[bidder][layer][k][0] != 0:
                            # add objective for s bound only if current bounds larger than zero
                            helper_Mip.maximize(self.s_help[(i, layer, k)])
                            helper_Mip.solve()
                            self.upper_bounds_s[bidder][layer][k][0] = np.ceil(self.s_help[(i, layer, k)].solution_value).astype(int)
                        del helper_Mip
        if verbose is True:
            logging.debug('Upper Bounds z:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
                logging.debug(v)
            logging.debug('\nUpper Bounds s:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_s.items()}).fillna('-').items():
                logging.debug(v)

    def print_upper_bounds(self, only_zeros=False):
        zeros = 0
        for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
            if not only_zeros:
                print('Upper Bounds z:')
                print(v)
                print()
            zeros = zeros + sum([np.sum(x == 0) for x in v])
        for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_s.items()}).fillna('-').items():
            if not only_zeros:
                print('Upper Bounds s:')
                print(v)
                print()
            zeros = zeros + sum([np.sum(x == 0) for x in v])
        print('Number of Upper bounds equal to 0: ', zeros)
# %%
print('MLCA NN_MIP Class imported')
