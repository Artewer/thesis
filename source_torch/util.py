#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file stores helper functions used across the files in this project.

"""

# Libs
import numpy as np
import random
import re
import logging
import pickle
from docplex.mp.model import Model
from scipy.spatial.distance import cdist
from numba import jit
from sklearn import linear_model
from torch.autograd import Variable
import torch
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from collections import OrderedDict

# get_source_mod = lambda : SourceModule('''
# __global__ void generate_combinations_cuda(int *matrix, int n)
# {
#     int idx = threadIdx.x + blockDim.x * blockIdx.x;
#     int tmp = idx;
#     for (int i = 0; i < n; i++) {
#         matrix[idx * n + i] = tmp % 2;
#         tmp /= 2;
#     }
# }
# ''')

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, params):
        super(linearRegression, self).__init__()
        self.params = params
        self.model = torch.nn.Linear(inputSize, outputSize)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=params['lr'])
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        out = self.model(x)
        return out

    def train(self, x, y):
        epochs = self.params['epochs']

        for epoch in range(epochs):
            inputs = Variable(torch.from_numpy(x))
            targets = Variable(torch.from_numpy(y))
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, targets)
            #print('epoch {}, loss {}'.format(epoch, loss.data[0]))
            print(epoch)
            print(loss)
            loss.backward()
            self.optimizer.step()

# %% (0) HELPER FUNCTIONS
# %%
def timediff_d_h_m_s(td):
    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return -(td.days), -int(td.seconds/3600), -int(td.seconds/60)%60, -(td.seconds%60)
    return td.days, int(td.seconds/3600), int(td.seconds/60)%60, td.seconds%60

# %% Tranforms bidder_key to integer bidder_id
# key = valid bidder_key (string), e.g. 'Bidder_0'
def key_to_int(key):
        return(int(re.findall(r'\d+', key)[0]))

# %% PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS for MLCA MECHANISM
# THIS METHOD USES TRUE UNIFORM SAMPLING!
# SATS_auction_instance = single instance of a value model
# number_initial_bids = number of initial bids
# bidder_ids = bidder ids in this value model (int)
# scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set


def _clean_weights(Wb):
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

def _get_model_weights(model): # torch
    nnmodel = model.model
    weights = []
    for params in nnmodel.parameters():
        weights.append(params.detach().cpu().numpy().T)        
    return weights



#value_model, bidder_id, n, presampled_n, ml_algorithm, presampled_algorithm='unif', ml_model = None):
def initial_bids_mlca_galo(SATS_auction_instance, number_initial_bids, bidder_names, presampled_n, presampled_algorithm, ml_model=None, scaler=None):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up initial Bids with GALO for: %s', bidder)
        D = galo_bids_init(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids, presampled_algorithm=presampled_algorithm, presampled_n=presampled_n, ml_model=ml_model)
        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)


def galo_bids_init(value_model, bidder_id, n, presampled_n, presampled_algorithm, ml_model):
    
    logging.debug('Sampling with GALO using 1' + str(presampled_algorithm))

    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    
    if presampled_algorithm == 'gali':
        D_presampled = gali_bids_init(value_model=value_model, bidder_id=bidder_id, n=presampled_n)
    else:
        D_presampled = unif_random_bids(value_model=value_model, bidder_id=bidder_id, n=presampled_n)
    
    #create a list of last values from D_presampled
    values = D_presampled[:, -1].tolist()
    bundles = D_presampled[:, :-1].tolist()
    M = len(value_model.get_good_ids())
    C = 6000
    # print('Initial bundles and values')
    # for bd in range(len(bundles)):
    #     print(str(bundles[bd]) + ' ' + str(values[bd]))

    count = 1
    while count <= (n-presampled_n):
        print(count)
        print('Here are the bundles')
        print(bundles)

        #Linear regression implementation
        #reg = linear_model.LinearRegression().fit(bundles, values)
        #reg = linear_model.LinearRegression(positive=True).fit(bundles, values)
        #reg = linear_model.Ridge(alpha=.5).fit(bundles, values)
        #coef = reg.coef_
        #intercept = reg.intercept_

        #Pytorch implementation
        # x_train = np.array(bundles).astype(np.float32)
        # y_train = np.array(values).astype(np.float32)
        # inputDim = x_train.shape[1] 
        # outputDim = 1
        # learningRate = 0.01
        # epochs = 1500
        # params = {'lr': 0.01, 'epochs': 1500}
        # model = linearRegression(inputDim, outputDim, params)
        # model.train(x_train, y_train)
        # W = _get_model_weights(model)
        # Wb = _clean_weights(W)
        # #create a list of weights and intercept
        # coef = Wb[0].flatten().tolist()
        # intercept = Wb[1].flatten().tolist()[0]

        #Linear Optimization
        # Sample data
        X = np.array(bundles) 
        #np.random.randint(100, size=(4, 10))

        y = np.array(values)
        #np.random.randint(100, size=(4, 1)).reshape(-1)

        n, p = X.shape

        # Create the model
        mdl = Model("LinearRegression")

        # Define the variables for regression coefficients
        beta = mdl.continuous_var_list(p, name="beta")

        # Define the quadratic objective
        XT_X = np.dot(X.T, X)
        XT_y = np.dot(X.T, y)
        obj = 0.5 * mdl.sum(XT_X[i][j] * beta[i] * beta[j] for i in range(p) for j in range(p)) - mdl.sum(XT_y[i] * beta[i] for i in range(p)) + 0.5 * np.dot(y, y)

        mdl.minimize(obj)


        # Solve the model
        mdl.solve()

        coef = [v.solution_value for v in beta]
        intercept = 0

        # Print the solution
        # for v in beta:
        #     print(f"{v.name} = {v.solution_value}")


        print('Here are trained coefficients and intercept')
        print(coef)
        print(intercept)
        x_vectors = []
        x_distances = []
        for i in range(len(bundles)):
            m = Model(name='sampling')

            r = m.continuous_var(name='r')
                        
            x = m.binary_var_list(range(M), name='x')
            
            b = m.binary_var_list(len(values), name='b')

            #Constraints

            #constraints = []

            y_prediction = m.sum(x[k]*coef[k] for k in range(M)) + intercept

            #constraints.append(m.sum(x[k]*coef[k] for k in range(M)) + intercept >= 0)
            m.add_constraint(m.sum(x[k]*coef[k] for k in range(M)) + intercept >= 0)

            m.add_constraint(y_prediction <= max(values))

            #constraints.append(values[i] - y_prediction <= r)
            m.add_constraint(values[i] - y_prediction <= r)

            #constraints.append(y_prediction - values[i] <= r)
            m.add_constraint(y_prediction - values[i] <= r)


            for j in range(len(values)):
                #constraints.append(values[j] - y_prediction + C * b[j] >= r)
                #constraints.append(y_prediction - values[j] + C * (1 - b[j]) >= r)
                m.add_constraint(values[j] - y_prediction + C * b[j] >= r)
                m.add_constraint(y_prediction - values[j] + C * (1 - b[j]) >= r)

            #m.add_constraints(constraints)

            m.maximize(r)

            sol = m.solve()

            try:
                distance = sol.get_objective_value()
                print('This is our objective value of r')
                print(sol.get_objective_value())
            except:
                print('No solution found')
                print(m.print_information())
                print(m.get_solve_status())
                print(m.get_solve_details())
                print(sol)
                #iterate over constraints
                for c in m.iter_constraints():
                    print(c)
            vector = np.array([x[k].solution_value for k in range(M)])
            x_vectors.append(vector)
            print('created vector')
            print(vector)
            distance1 = np.sum(np.abs(np.sum(vector*coef) + intercept - values[i]))
            print('distance for this vector*coefficient + intercept - its true value. Now we use r for distance')
            print(distance1)
            x_distances.append(distance)
        chosen_bundle = x_vectors[np.argmax(x_distances)]
        value = myfunc(chosen_bundle)
        values.append(value)
        bundles.append(chosen_bundle.tolist())
        print('Here is the solution vector and its value')
        print(chosen_bundle)
        print(value)
        count+=1
    del myfunc
    D = np.array(bundles)
    D = np.hstack((D, np.array(values).reshape(-1, 1)))
    return (D)




def initial_bids_mlca_gali(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up initial Bids with GALI Learning Method for: %s', bidder)
        D = gali_bids_init(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)
        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]
    if scaler is not None: 
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)


def gali_bids_init(value_model, bidder_id, n):
    logging.debug('Sampling with GALI %s bundle-value pairs from bidder %s',n, bidder_id)
    M = len(value_model.get_good_ids())  # number of items in value model
    b_0 = np.asarray(np.random.choice([0, 1], size=M))
    S_matrix = []
    S_matrix.append(b_0)
    #S_matrix = np.zeros((n, M))  # initialize the matrix directly
    #S_matrix[0] = np.random.choice([0, 1], size=M)  # set the first row
    for i in range(n-1):
        n_S_matrix = len(S_matrix)
        x_vectors = []
        x_distances = []
        for j in range(n_S_matrix):
            m = Model(name='greedy sampling')
            indices = range(M)
            # 2) initiate bundle we want to find, bundle that we compare right now and other sampled bundles
            x = m.binary_var_list(keys = M, name='x')

            #Define objective function
            obj_function = m.sum(x[i] + S_matrix[j][i] - 2 * x[i] * S_matrix[j][i] for i in indices)

            constraints = []
            for k in range(n_S_matrix):
                if k == j:
                    continue
                constraint_expr = m.sum(x[i] + S_matrix[k][i] - 2 * x[i] * S_matrix[k][i] for i in indices)
                constraints.append(obj_function <= constraint_expr)
            m.add_constraints(constraints)

            m.maximize(obj_function)
            m.solve()
            vector = np.array([x[i].solution_value for i in range(M)])
            #Calculate manhattan distance between bundle and corresponding bundle in S_matrix
            distances = np.sum(np.abs(vector - S_matrix[j]))            
            x_vectors.append(vector)
            x_distances.append(distances)
        S_matrix.append(x_vectors[np.argmax(x_distances)])
    D = np.array(S_matrix)
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return (D)



def initial_bids_mlca_active_learning(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up initial Bids with Active Learning Method for: %s', bidder)
        D = active_learning_bids_init(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)
        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]

    if scaler is not None: 
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)



def gali_bids_init(value_model, bidder_id, n):
    logging.debug('Sampling with GALI %s bundle-value pairs from bidder %s',n, bidder_id)
    M = len(value_model.get_good_ids())  # number of items in value model
    b_0 = np.asarray(np.random.choice([0, 1], size=M))
    S_matrix = []
    S_matrix.append(b_0)
    #S_matrix = np.zeros((n, M))  # initialize the matrix directly
    #S_matrix[0] = np.random.choice([0, 1], size=M)  # set the first row
    for i in range(n-1):
        n_S_matrix = len(S_matrix)
        x_vectors = []
        x_distances = []
        for j in range(n_S_matrix):
            m = Model(name='greedy sampling')
            indices = range(M)
            # 2) initiate bundle we want to find, bundle that we compare right now and other sampled bundles
            x = m.binary_var_list(keys = M, name='x')

            #Define objective function
            obj_function = m.sum(x[i] + S_matrix[j][i] - 2 * x[i] * S_matrix[j][i] for i in indices)

            constraints = []
            for k in range(n_S_matrix):
                if k == j:
                    continue
                constraint_expr = m.sum(x[i] + S_matrix[k][i] - 2 * x[i] * S_matrix[k][i] for i in indices)
                constraints.append(obj_function <= constraint_expr)
            m.add_constraints(constraints)

            m.maximize(obj_function)
            m.solve()
            vector = np.array([x[i].solution_value for i in range(M)])
            #Calculate manhattan distance between bundle and corresponding bundle in S_matrix
            distances = np.sum(np.abs(vector - S_matrix[j]))            
            x_vectors.append(vector)
            x_distances.append(distances)
        S_matrix.append(x_vectors[np.argmax(x_distances)])
    D = np.array(S_matrix)
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return (D)



def initial_bids_mlca_active_learning(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up initial Bids with Active Learning Method for: %s', bidder)
        D = active_learning_bids_init(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)
        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]

    if scaler is not None: 
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)


def active_learning_bids_init(value_model, bidder_id, n):
    logging.debug('Sampling with active learning %s bundle-value pairs from bidder %s',n, bidder_id)
    M = len(value_model.get_good_ids())  # number of items in value model
    b_0 = np.asarray(np.random.choice([0, 1], size=M))
    S_matrix = []
    S_matrix.append(b_0)
    for i in range(n-1):
        n_S_matrix = len(S_matrix)
        x_vectors = []
        x_distances = []
        for j in range(n_S_matrix):
            m = Model(name='greedy sampling')
            indices = range(M)
            # 2) initiate bundle we want to find, bundle that we compare right now and other sampled bundles
            x = m.binary_var_list(keys = M, name='x')

            #Define objective function
            obj_function = m.sum(x[i] + S_matrix[j][i] - 2 * x[i] * S_matrix[j][i] for i in indices)

            for k in range(n_S_matrix):
                if k == j:
                    continue
                constraint_expr = m.sum(x[i] + S_matrix[k][i] - 2 * x[i] * S_matrix[k][i] for i in indices)
                m.add_constraint(obj_function <= constraint_expr)

            m.maximize(obj_function)
            m.solve()
            vector = np.array([x[i].solution_value for i in range(M)])
            distances = np.sum(np.sum(np.abs(S_matrix-vector))) #TODO test this
            m = Model(name='greedy sampling')
            indices = range(M)
            # 2) initiate bundle we want to find, bundle that we compare right now and other sampled bundles
            x = m.binary_var_list(keys = M, name='x')

            #Define objective function
            obj_function = m.sum(x[i] + S_matrix[j][i] - 2 * x[i] * S_matrix[j][i] for i in indices)

            for k in range(n_S_matrix):
                if k == j:
                    continue
                constraint_expr = m.sum(x[i] + S_matrix[k][i] - 2 * x[i] * S_matrix[k][i] for i in indices)
                m.add_constraint(obj_function <= constraint_expr)

            m.maximize(obj_function)
            m.solve()
            vector = np.array([x[i].solution_value for i in range(M)])
            distances = np.sum(np.sum(np.abs(S_matrix-vector.reshape(1, M)),axis=0))
            x_vectors.append(vector)
            x_distances.append(distances)
        S_matrix.append(x_vectors[np.argmax(x_distances)])
    D = np.array(S_matrix)
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return (D)



def generate_combinations_pycuda(n):
    mod = get_source_mod()
    print(mod)
    generate_combinations_cuda = mod.get_function("generate_combinations_cuda")
    matrix_gpu = np.zeros((2**n, n), dtype=np.int32)
    block_size = 32
    grid_size = (2**n + block_size - 1) // block_size
    generate_combinations_cuda(drv.Out(matrix_gpu), np.int32(n), block=(block_size, 1, 1), grid=(grid_size,1))
    return matrix_gpu


def generate_combinations(n):
    # create an empty list to store the combinations
    combinations = []
    
    # generate all possible combinations of 0's and 1's for n items
    for i in range(2**n):
        # convert i to a binary string of length n with leading zeros
        binary_str = bin(i)[2:].zfill(n)
        
        # convert the binary string to a list of integers
        combination = [int(digit) for digit in binary_str]
        
        # append the combination to the list of combinations
        combinations.append(combination)
    return combinations

def initial_bids_mlca_fft_reverse(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up initial Bids with Farthest First Traversal Reverse for: %s', bidder)
        D = fft_reverse_bids_init(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)
        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]
    #TODO what is scaler is not None
    if scaler is not None: #TODO Next iterations
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)

def fft_reverse_bids_init(value_model, bidder_id, n):
    logging.debug('Sampling with ffd_reverse at random %s bundle-value pairs from bidder %s',n, bidder_id)
    ncol = len(value_model.get_good_ids())  # number of items in value model
    b_0 = np.asarray(np.random.choice([0, 1], size=ncol))
    #Empty array to store chosen bids
    D = np.empty((0, ncol))
    # Add starting bid to chosen bids
    D = np.append(D, b_0.reshape(1, -1), axis=0)
    for i in range(n-2):
        #Calculate the reverse bid for each bid in D 
        reverse_bids = np.logical_not(D)
        #Average reverse bidsss
        reverse_bids = np.mean(reverse_bids, axis=0)
        #Calculate for every bid which value is closer to the average reverse bid 0 or 1. If value is 0.5 randomize it
        for j in range(len(reverse_bids)):
            if reverse_bids[j] == 0.5:
                reverse_bids[j] = random.randint(0,1)
            else:
                reverse_bids[j] = np.round(reverse_bids[j])
        #Add the reverse bid to the chosen bids
        D = np.append(D, reverse_bids.reshape(1, -1), axis=0)
    # define helper function for specific bidder_id
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return (D)

def initial_bids_mlca_fft(SATS_auction_instance, number_initial_bids, bidder_names, bundle_number=10, scaler=None):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up initial Bids with Farthest First Traversal for: %s', bidder)
        D = fft_bids_init(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)
        empty = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, empty, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]
    if scaler is not None: #TODO Next iterations
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)

def fft_bids_init(value_model, bidder_id, n):
    logging.debug('Sampling with ffd at random %s bundle-value pairs from bidder %s',n, bidder_id)
    ncol = len(value_model.get_good_ids())  # number of items in value model
    # generate random starting bid uniformly
    available_bids = generate_combinations(ncol)
    b_0 = available_bids[np.random.randint(0, len(available_bids))]
    D = np.zeros((n, ncol))
    D[0] = b_0
    for i in range(1, n-1):
        distances = cdist(available_bids, D[:i], metric='euclidean')
        sum_dist = np.sum(distances, axis=1)
        D[i] = available_bids[np.argmax(sum_dist)]
    # for i in range(n-1):
    #     sum_distance = np.zeros(len(available_bids))
    #     # Calculate euclidean distance between chosen bids and all available bids
    #     for D_i in D:
    #         D_i = D_i[np.newaxis, :]
    #         distance = cdist(available_bids, D_i, metric='euclidean')
    #         sum_distance += distance.flatten()
    #     # Find an availabel bid with the maximum distance to all chosen bids
    #     answers = np.argwhere(sum_distance == np.amax(sum_distance))
    #     #if bid is already in the chosen bundle of bids -> choose the next one
    #     # for a in answers:
    #     #     if available_bids[a] not in D:
    #     #         D = np.append(D, available_bids[a].reshape(1, -1), axis=0)
    #     #         break
    #     #     #if a is last element in answers and doesn't fit -> uniform sampling
    #     #     elif a == answers[-1]:
    #     #         D = np.append(D, available_bids[random.randint(0,len(available_bids)-1)].reshape(1, -1), axis=0)
    #     # Add the bid with the maximum distance to the chosen bids
    #     D = np.append(D, available_bids[answers[0]].reshape(1, -1), axis=0)
    #     #D = np.append(D, available_bids[max_dist_idx].reshape(1, -1), axis=0)
    # # define helper function for specific bidder_id
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return (D)


#value_model, bidder_id, n, presampled_n, ml_algorithm, presampled_algorithm='unif', ml_model = None):
def initial_bids_mlca_unif(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None,):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up initial Bids for: %s', bidder)
        #D = unif_random_bids((SATS_auction_instance, number_initial_bids, bidder_names)
        D = unif_random_bids(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)
        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)

# %% PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS
# THIS METHOD USES TRUE UNIFORM SAMPLING!
# value_model = single instance of a value model
# c0 = number of initial bids
# bidder_ids = bidder ids in this value model (int)
# scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set


def initial_bids_pvm_unif(value_model, c0, bidder_ids, scaler=None):
    initial_bids = OrderedDict()
    for bidder_id in bidder_ids:
        logging.debug('Set up intial Bids for: Bidder_{}'.format(bidder_id))
        D = unif_random_bids(value_model=value_model, bidder_id=bidder_id, n=c0)
        # add zero bundle
        null = np.zeros(D.shape[1]).reshape(1, -1)
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids['Bidder_{}'.format(bidder_id)] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder_id in bidder_ids:
            tmp = np.concatenate((tmp, initial_bids['Bidder_{}'.format(bidder_id)][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return([initial_bids, scaler])
# %% PREPARE INITIAL BIDS FOR A SINGLE INSTANCE FOR ALL BIDDERS
# THIS METHOD USES RANDOM SAMPLING OF BUNDLES FROM SATS VIA NORMAL DISTRIBUTION!
# value_model = single instance of a value model
# c0 = number of initial bids
# bidder_ids = bidder ids in this value model (int)
# scaler = scale the y values across all bidders, fit on the selected training set and apply on the validation set
# seed = seed corresponding to the bidder_ids


def initial_bids_pvm(value_model, c0, bidder_ids, scaler=None, seed=None):
    initial_bids = OrderedDict()
    for bidder_id in bidder_ids:
        logging.debug('Set up intial Bids for: Bidder_{}'.format(bidder_id))
        if seed is not None:
            D = np.array(value_model.get_random_bids(bidder_id=bidder_id, number_of_bids=c0, seed=seed[bidder_id]))
        else:
            D = np.array(value_model.get_random_bids(bidder_id=bidder_id, number_of_bids=c0))
        # add zero bundle
        null = np.zeros(D.shape[1]).reshape(1, -1)
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids['Bidder_{}'.format(bidder_id)] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder_id in bidder_ids:
            tmp = np.concatenate((tmp, initial_bids['Bidder_{}'.format(bidder_id)][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s', scaler.scale_, ' | ', float(scaler.data_max_ * scaler.scale_), '== feature range max?')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return([initial_bids, scaler])

# %% This function formates the solution of the winner determination problem (WDP) given elicited bids.
# Mip = A solved DOcplex instance.
# elicited_bids = the set of elicited bids for each bidder corresponding to the WDP.
# bidder_names = bidder names (string, e.g., 'Bidder_1')
# fitted_scaler = the fitted scaler used in the valuation model.

def format_solution_mip_new(Mip, elicited_bids, bidder_names, fitted_scaler):
    tmp = {'good_ids': [], 'value': 0}
    Z = OrderedDict()
    for bidder_name in bidder_names:
        Z[bidder_name] = tmp
    S = Mip.solution.as_dict()
    for key in list(S.keys()):
        index = [int(x) for x in re.findall(r'\d+', key)]
        bundle = elicited_bids[index[0]][index[1], :-1]
        value = elicited_bids[index[0]][index[1], -1]
        if fitted_scaler is not None:
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug(value)
            logging.debug('WDP values for allocation scaled by: 1/%s',round(fitted_scaler.scale_[0],8))
            value = float(fitted_scaler.inverse_transform([[value]]))
            logging.debug(value)
            logging.debug('---------------------------------------------')
        bidder = bidder_names[index[0]]
        Z[bidder] = {'good_ids': list(np.where(bundle == 1)[0]), 'value': value}
    return(Z)

# %% This function generates bundle-value pairs for a single bidder sampled uniformly at random from the bundle space.
# value_model = SATS auction model instance generated via PySats
# bidder_id = bidder id (int)
# n = number of bundle-value pairs.

def unif_random_bids(value_model, bidder_id, n):
    logging.debug('Sampling uniformly at random %s bundle-value pairs from bidder %s',n, bidder_id)
    ncol = len(value_model.get_good_ids())  # number of items in value model
    D = np.unique(np.asarray(random.choices([0,1], k=n*ncol)).reshape(n,ncol), axis=0)
    # get unique ones if accidently sampled equal bundle
    while D.shape[0] != n:
        tmp = np.asarray(random.choices([0,1], k=ncol)).reshape(1, -1)
        D = np.unique(np.vstack((D, tmp)), axis=0)
    # define helper function for specific bidder_id
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    D = np.hstack((D, np.apply_along_axis(myfunc, 1, D).reshape(-1, 1)))
    del myfunc
    return (D)


#########################################################################################################################
# new functions

#this function initialize query uniformly but also ensures 1 column 0

def unif_random_bids_zero(value_model, bidder_id, n):
    logging.debug('Sampling uniformly at random %s bundle-value pairs from bidder %s',n, bidder_id)
    ncol = len(value_model.get_good_ids())  # number of items in value model
    D = np.unique(np.asarray(random.choices([0,1], k=n*ncol)).reshape(n,ncol), axis=0)
    # get unique ones if accidently sampled equal bundle
    while D.shape[0] != n:
        tmp = np.asarray(random.choices([0,1], k=ncol)).reshape(1, -1)
        D = np.unique(np.vstack((D, tmp)), axis=0)
    Z = make_sure_zero(D) #here mthis makes one zero column
    # define helper function for specific bidder_id
    def myfunc(bundle):
        return value_model.calculate_value(bidder_id, bundle)
    Z = np.hstack((Z, np.apply_along_axis(myfunc, 1, Z).reshape(-1, 1)))
    del myfunc
    return(Z)

# this functions saves the result
# dir_file is combination of directory to save and file name
# result is the result of experiment

def save_result(dir_file,result):
    with open(dir_file, 'wb') as f:
        pickle.dump(result, f)
        f.close()
    return('Result is saved as'+dir_file)

# this functions loads the result
# main purpose is for plotting and creating summary over all the experiments

def load_result(dir_file):
    with open(dir_file, 'rb') as f:
        result = pickle.load(f)
        f.close()
        print(dir_file+' is loaded ')
    return(result)


# this function is needed for good initialization which gurantees randomness+non-similarity+no empty column
# Qinit
# bidder_names
# value model must be the initialized SATS model

def initial_bids_mlca_extra(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None):
    initial_bids_rand = OrderedDict()
    for bidder_id in bidder_names:
        logging.debug('Set up intial Bids for: %s', bidder_id)
        logging.debug('Sampling uniformly at random %s bundle-value pairs from bidder %s', number_initial_bids, bidder_id)
        bidder = key_to_int(bidder_id)
        ncol = len(SATS_auction_instance.get_good_ids())  # number of items in value model

        Z=np.zeros(shape=(number_initial_bids,ncol))
        for row in range(number_initial_bids):
            Z[row, np.random.choice(np.arange(ncol), size=random.choices(np.arange(1, ncol)), replace=False)] = 1
            # this is used for init with few items in each bundle
            # Z[row, np.random.choice(np.arange(ncol), size=random.choices(np.arange(1, number_initial_bids)), replace=False)] = 1


    # get unique ones if accidently sampled equal bundle
        while np.unique(Z, axis=0).shape[0] != number_initial_bids:
            logging.debug('Non unique row founded for %s', bidder_id)
            tmp = np.asarray(random.choices([0, 1], k=ncol)).reshape(1, -1)
            Z = np.unique(np.vstack((Z, tmp)), axis = 0)

    # check if all items are in one bundle and randomly generate a column for that item
        sum_k= np.zeros(shape=ncol)
        for i in range (ncol):
            sum_k[i]= np.sum(Z[:, i])
        if any(sum_k ==0 ):
            logging.debug('There is an item that is not in bundle for %s',bidder_id)
            l= np.argwhere(sum_k==0)
            for i in range(len(l)):
                Z[(random.choices(np.arange(number_initial_bids), k=random.choice(np.arange(1, number_initial_bids)))), l[i]] = 1

        # define helper function for specific bidder_id
        def myfunc(bundle):
            return SATS_auction_instance.calculate_value(bidder, bundle)
        Z = np.hstack((Z, np.apply_along_axis(myfunc, 1, Z).reshape(-1, 1)))
        del myfunc
        # add zero bundle
        null = np.zeros(Z.shape[1]).reshape(1, -1)
        Z = np.append(Z, null, axis=0)

    #   Z = np.unique(Z,axis=0) this sorts bundles in increasing order
        X = Z[:, :-1]
        Y = Z[:, -1]
        initial_bids_rand[bidder_id] = [X, Y]
        ##this apperantly not working for mrvm
        if scaler is not None:
            tmp = np.array([])
            for bidder in bidder_names:
                # bidder_id = key_to_int(bidder)
                tmp = np.concatenate((tmp, initial_bids_rand[bidder_id][1]), axis=0)
            scaler.fit(tmp.reshape(-1, 1))
            logging.debug('')
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug('Samples seen: %s', scaler.n_samples_seen_)
            logging.debug('Data max: %s', scaler.data_max_)
            logging.debug('Data min: %s', scaler.data_min_)
            logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_,
                          float(scaler.data_max_ * scaler.scale_))
            logging.debug('---------------------------------------------')
            initial_bids_rand = OrderedDict(list(
                (key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in
                initial_bids_rand.items()))

    return(initial_bids_rand, scaler)


##mlca unif with zero in one column

def initial_bids_mlca_unif_empty(SATS_auction_instance, number_initial_bids, bidder_names, scaler=None):
    initial_bids = OrderedDict()
    for bidder in bidder_names:
        logging.debug('Set up intial Bids for: %s', bidder)
        D = unif_random_bids_zero(value_model=SATS_auction_instance, bidder_id=key_to_int(bidder), n=number_initial_bids)
        null = np.zeros(D.shape[1]).reshape(1, -1) # add null bundle
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids[bidder] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder in bidder_names:
            tmp = np.concatenate((tmp, initial_bids[bidder][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('')
        logging.debug('*SCALING*')
        logging.debug('---------------------------------------------')
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        logging.debug('---------------------------------------------')
        initial_bids = OrderedDict(list((key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in initial_bids.items()))
    return(initial_bids, scaler)


##pvm with unif and empty zero column

def initial_bids_pvm_unif_empty(value_model, c0, bidder_ids, scaler=None):
    initial_bids = OrderedDict()
    for bidder_id in bidder_ids:
        logging.debug('Set up intial Bids for: Bidder_{}'.format(bidder_id))
        D = unif_random_bids_zero(value_model=value_model, bidder_id=bidder_id, n=c0)
        # add zero bundle
        null = np.zeros(D.shape[1]).reshape(1, -1)
        D = np.append(D, null, axis=0)
        X = D[:, :-1]
        Y = D[:, -1]
        initial_bids['Bidder_{}'.format(bidder_id)] = [X, Y]
    if scaler is not None:
        tmp = np.array([])
        for bidder_id in bidder_ids:
            tmp = np.concatenate((tmp, initial_bids['Bidder_{}'.format(bidder_id)][1]), axis=0)
        scaler.fit(tmp.reshape(-1, 1))
        logging.debug('Samples seen: %s', scaler.n_samples_seen_)
        logging.debug('Data max: %s', scaler.data_max_)
        logging.debug('Data min: %s', scaler.data_min_)
        logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_, float(scaler.data_max_ * scaler.scale_))
        initial_bids = OrderedDict(list(
            (key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in
            initial_bids.items()))
    return (initial_bids, scaler)

##pvm that gurantees full columns

def initial_bids_pvm_extra(value_model, c0, bidder_ids, scaler=None):
    initial_bids_rand = OrderedDict()
    for bidder_id in bidder_ids:
        logging.debug('Set up intial Bids for: %s', bidder_id)
        logging.debug('Sampling uniformly at random %s bundle-value pairs from bidder %s', c0, bidder_id)
        ncol = len(value_model.get_good_ids())  # number of items in value model

        Z=np.zeros(shape=(c0,ncol))
        for row in range(c0):
            Z[row, np.random.choice(np.arange(ncol), size=random.choices(np.arange(1, ncol)), replace=False)] = 1

    # get unique ones if accidently sampled equal bundle
        while np.unique(Z, axis=0).shape[0] != c0:
            logging.debug('Non unique row founded for %s', bidder_id)
            tmp = np.asarray(random.choices([0, 1], k=ncol)).reshape(1, -1)
            Z = np.unique(np.vstack((Z, tmp)), axis = 0)

    # check if all items are in one bundle and randomly generate a column for that item
        sum_k= np.zeros(shape=ncol)
        for i in range (ncol):
            sum_k[i]= np.sum(Z[:, i])
        if any(sum_k ==0 ):
            logging.debug('There is an item that is not in bundle for %s',bidder_id)
            l= np.argwhere(sum_k==0)
            for i in range(len(l)):
                Z[(random.choices(np.arange(c0), k=random.choice(np.arange(1, c0)))), l[i]] = 1

        # define helper function for specific bidder_id
        def myfunc(bundle):
            return value_model.calculate_value(bidder_id, bundle)
        Z = np.hstack((Z, np.apply_along_axis(myfunc, 1, Z).reshape(-1, 1)))
        del myfunc
        # add zero bundle
        null = np.zeros(Z.shape[1]).reshape(1, -1)
        Z = np.append(Z, null, axis=0)

    #   Z = np.unique(Z,axis=0) this sorts bundles in increasing order
        X = Z[:, :-1]
        Y = Z[:, -1]
        initial_bids_rand['Bidder_{}'.format(bidder_id)] = [X, Y]
        ##this apperantly not working for mrvm
        if scaler is not None:
            tmp = np.array([])
            for bidder in bidder_ids:
                tmp = np.concatenate((tmp, initial_bids_rand[bidder][1]), axis=0)
            scaler.fit(tmp.reshape(-1, 1))
            logging.debug('')
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug('Samples seen: %s', scaler.n_samples_seen_)
            logging.debug('Data max: %s', scaler.data_max_)
            logging.debug('Data min: %s', scaler.data_min_)
            logging.debug('Scaling by: %s | %s==feature range max?', scaler.scale_,
                          float(scaler.data_max_ * scaler.scale_))
            logging.debug('---------------------------------------------')
            initial_bids_rand = OrderedDict(list(
                (key, [value[0], scaler.transform(value[1].reshape(-1, 1)).flatten()]) for key, value in
                initial_bids_rand.items()))

    return(initial_bids_rand, scaler)

## gets the column which has least 1 and makes that column 0, if there are more columns chooses with a probability.

def make_sure_zero(Z):
    sum_k= np.zeros(Z.shape[0])
    for i in range (Z.shape[0]):
        sum_k[i]= np.sum(Z[:, i])
    if any(sum_k ==0 ):
        l= np.argwhere(sum_k==0)
        print('the column is 0')

    else:
        l = np.where(sum_k == np.amin(sum_k))
        _index = np.random.choice(l[0], 1)
        Z[:,_index]=0
    return(Z)