# Cross-validation tools
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/klow12/sn29/klow/ML/DBs/IEcalcs/mixed/codes/elec-dens/all_copied/')
import time
from sklearn.model_selection import RepeatedKFold
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from scipy.linalg import cholesky, solve_triangular
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from comparative_normalised import ElementalDimerSumKernel, ElementalDimerAverageKernel, ElementalSOAPKernel
from dscribe.kernels import REMatchKernel, AverageKernel, SumKernel
import os 
import matplotlib.pyplot as plt
import warnings
from itertools import repeat

def rmse_percent(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    loss = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    return loss*100

def get_metrics(y_true, y_pred):
    d = {}
    d['rmse'] = np.round(np.sqrt(mean_squared_error(y_true, y_pred)),2)
    d['%rmse'] = np.round(rmse_percent(y_true, y_pred), 2)
    d['mae'] = np.round(mean_absolute_error(y_true, y_pred),2)
    d['r2'] = np.round(r2_score(y_true, y_pred),2)
    return d

class hybrid_kernel_CV(BaseEstimator):
    def __init__(self, alpha=0.5, rematch=0.5, degree=2, krr_param_grid=None, cv_nfolds=5, cv_shuffles=1, 
    verbose=True, copy_X=True, mae=False, print_errors=False, normalize_kernel=True):
        self.krr_param_grid = krr_param_grid # Grid of: gamma values, possible kernel funcs
        self.normalize_kernel = normalize_kernel
        self.alpha=alpha
        self.rematch = rematch
        self.degree = degree
        self.global_kernel1 = ElementalDimerAverageKernel
        self.global_kernel2 = ElementalSOAPKernel
        self.cv_nfolds = cv_nfolds
        self.verbose = verbose
        self.cv_shuffles = cv_shuffles
        self.copy_X = copy_X
        self.mae = mae
        self.print_errors = print_errors

    def fit(self, X, y, natoms, atomtypes, cv_indices=None, random_state=None):
        '''
        Fits the model using n-fold cross-validation.
            natoms (list): list of number of atoms in the first monomer (inefficient, I know)
            atomtypes (array of lists): element of each atom in the dimer
            cv_indices (array): option to provide own predefined CV splits.
            show_plot (bool): whether to show heatmap of CV values or not.
        '''
        t = time.time()

        if y.ndim < 2:
            y = y.reshape(-1,1)

        if cv_indices is None:
            cv_indices = np.arange(X.shape[0]) 

        kfold = RepeatedKFold(n_splits=self.cv_nfolds, n_repeats=self.cv_shuffles, random_state=random_state)
        cv_folds = kfold.split(X[cv_indices])
        n_cv_folds = kfold.get_n_splits()

        add_train_inds = np.setdiff1d(np.arange(X.shape[0]), cv_indices) # Empty if cv_indices was None
        cv_folds = list(cv_folds)
        cv_folds = [(np.concatenate((train_fold, add_train_inds)), test_fold) for train_fold, test_fold in cv_folds]
        # Returns list of (train folds, test folds) * n_cv

        if self.verbose > 0:
            elapsed = time.time() - t
            print('Starting cross-validation [%dmin %dsec]' % (int(elapsed/60), int(elapsed%60)))
            sys.stdout.flush()

        y_ = y
        errors = []

        for fold_i, (train_i, test_i) in enumerate(cv_folds):
            fold_errors = np.empty((len(self.krr_param_grid['gamma']),
                                    len(self.krr_param_grid['kernel_func']),
                                    len(self.krr_param_grid['alpha']),
                                    len(self.krr_param_grid['lambda']),
                                    y_.shape[1]))
            if self.verbose > 0:
                elapsed = time.time() - t
                print('CV %d of %d [%dmin %dsec]' % (fold_i + 1,
                                                        n_cv_folds,
                                                        int(elapsed / 60),
                                                        int(elapsed % 60)))
                sys.stdout.flush()
            
            
            for gamma_i, gamma in enumerate(self.krr_param_grid['gamma']):
                if self.verbose > 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                
                for func_i, func in enumerate(self.krr_param_grid['kernel_func']):
                    for alpha_i, alpha in enumerate(self.krr_param_grid['alpha']):
                        kernel1 = self.global_kernel1(metric=func, gamma=gamma, normalize_kernel=self.normalize_kernel)
                        kernel2 = self.global_kernel2(degree=self.degree, normalize_kernel=self.normalize_kernel, alpha=self.rematch)
                        
                        X_train, X_test = X[train_i], X[test_i]
                        train_atoms, test_atoms = natoms[train_i], natoms[test_i]
                        train_atomtypes, test_atomtypes = atomtypes[train_i], atomtypes[test_i]

                        K_train1 = kernel1.create(X_train, train_atoms, train_atomtypes)
                        K_train2 = kernel2.create(X_train, train_atoms, train_atomtypes)
                        K_train = (1-alpha)*K_train1 + alpha*K_train2

                        K_test1 = kernel1.create(X_test, test_atoms, test_atomtypes, X_train, train_atoms, train_atomtypes)
                        K_test2 = kernel2.create(X_test, test_atoms, test_atomtypes, X_train, train_atoms, train_atomtypes)
                        K_test = (1-alpha)*K_test1 + alpha*K_test2

                        for lamb_i, lamb in enumerate(self.krr_param_grid['lambda']):
                            y_train, y_test = y_[train_i], y_[test_i]
                            y_scaler = StandardScaler()
                            y_train = y_scaler.fit_transform(y_train)

                            if self.verbose > 0:
                                sys.stdout.write(',')
                                sys.stdout.flush()
                            K_train_ = K_train.copy()
                            K_train_.flat[::K_train_.shape[0] + 1] += lamb
                            try:
                                
                                L_ = cholesky(K_train_, lower=True)
                                x = solve_triangular(L_, y_train, lower=True)
                                dual_coef_ = solve_triangular(L_.T, x)
                                pred_mean = np.dot(K_test, dual_coef_)
                                pred_mean = y_scaler.inverse_transform(pred_mean)
                                if self.mae:
                                    e = np.mean(np.abs(pred_mean - y_test), 0)
                                else: # return RMSE
                                    e = np.sqrt(np.mean(((pred_mean - y_test) ** 2), 0))
                            except np.linalg.LinAlgError:
                                e = np.inf
                            fold_errors[gamma_i, func_i, alpha_i, lamb_i] = e
            
            if self.verbose > 0:
                sys.stdout.write('\n')
                sys.stdout.flush()
            errors.append(fold_errors)
        errors = np.array(errors)
        
        if self.print_errors:
                print('CV errors: ', errors.tolist())
                print('Average errors: ', np.mean(errors,0))
        errors = np.mean(errors, 0)  # average over folds

        self.dual_coefs_ = np.empty((y_.shape[1], X.shape[0]))
        self.lambs_ = np.empty(y_.shape[1])
        self.alphas_ = np.empty(y_.shape[1])
        self.kernel_func_ = np.empty(y_.shape[1])
        self.gammas_ = np.empty(y_.shape[1])
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Refit [%dmin %dsec]' % (int(elapsed/60), int(elapsed%60)))
            sys.stdout.flush()
        # single_combo = True, means can automatically increase lambda (regularisation) value

        errors = np.mean(errors, -1)
        if self.verbose > 0:
            print('Gamma params: ', self.krr_param_grid['gamma'])
            print('Kernel functions: ', self.krr_param_grid['kernel_func'])
            print('Lambda params: ', self.krr_param_grid['lambda'])
            print('Alpha params: ', self.krr_param_grid['alpha'])

        min_params = np.argsort(errors, axis=None)
        gamma_i, kernel_func_i, alpha_i, lambda_i = np.unravel_index(min_params[0], errors.shape)

        gamma = self.krr_param_grid['gamma'][gamma_i]
        kernel_func = self.krr_param_grid['kernel_func'][kernel_func_i]
        alpha = self.krr_param_grid['alpha'][alpha_i]
        lamb = self.krr_param_grid['lambda'][lambda_i]

        print('CV results: gamma={}, lambda={}, kernel_func={}, alpha={:.1f}, RMSE={:.3f}'.format(gamma, 
                lamb, kernel_func, alpha, np.min(errors)))

        if len(self.krr_param_grid['gamma']) > 1:
            if gamma == self.krr_param_grid['gamma'][0]:
                print('Gamma at lower edge.')
            if gamma == self.krr_param_grid['gamma'][-1]:
                print('Gamma at upper edge.')
        if len(self.krr_param_grid['lambda']) > 1:
            if lamb == self.krr_param_grid['lambda'][0]:
                print('Lambda at lower edge.')
            if lamb == self.krr_param_grid['lambda'][-1]:
                print('Lambda at upper edge.')
        
        self.dual_coefs_ = np.empty((y_.shape[1], X.shape[0]))
        self.gammas_[:] = gamma
        self.kernel_func_ = kernel_func
        self.alphas_[:] = alpha
        self.lambs_[:] = lamb

        combos = list(zip(self.gammas_, self.alphas_, self.lambs_, repeat(self.kernel_func_)))
         # if more than one optimal set of gamma/lambda, but use only one kernel_func
        n_unique_combos = len(set(combos))
        self.L_fit_ = [None] * n_unique_combos
        for i, (gamma, alpha, lamb, func) in enumerate(set(combos)):
            if self.verbose > 0:
                elapsed = time.time() - t
                print('Parameter combinations ' +  '%d of %d [%dmin %dsec]' % (
                                            i+1, n_unique_combos,
                                            int(elapsed/60),int(elapsed%60)))
            y_list = [i for i in range(y_.shape[1]) if
                self.lambs_[i] == lamb and self.gammas_[i] == gamma and self.alphas_[i] == alpha and self.kernel_func_ == kernel_func]
            
            # Refit to the newly determined parameters 
            self.final_kernel1 = self.global_kernel1(metric=kernel_func, gamma=gamma, normalize_kernel=self.normalize_kernel)
            self.final_kernel2 = self.global_kernel2(degree=self.degree, normalize_kernel=self.normalize_kernel, alpha=self.rematch)
            K1 = self.final_kernel1.create(X, natoms, atomtypes)
            K2 = self.final_kernel2.create(X, natoms, atomtypes)
            K = (1-alpha)*K1 + alpha*K2
            self.y_all_scaler = StandardScaler()
            y_all = self.y_all_scaler.fit_transform(y_[:, y_list])

            while True: 
                K.flat[::K.shape[0] + 1] += lamb - (lamb/10)
                try:
                    if self.verbose > 0:
                        print('Trying Cholesky decomposition, lambda = ', lamb)
                    L_ = cholesky(K, lower=True)
                    self.L_fit_[i] = L_
                    x = solve_triangular(L_, y_all, lower=True)
                    dual_coef_ = solve_triangular(L_.T, x)
                    self.dual_coefs_[y_list] = dual_coef_.T.copy()
                    break
                except np.linalg.LinAlgError:
                    if self.verbose > 0:
                        print('Linalg error, increasing lambda.')
                    lamb *= 10
                    self.lambs_[0] = lamb

        if self.copy_X:
            self.X_fit_ = X.copy()
            self.natoms_ = natoms.copy()
            self.atomtypes_ = atomtypes.copy()
            self.y_fit_ = y.copy()
        else:
            self.X_fit_ = X
            self.y_fit_ = y
            self.natoms_ = natoms
            self.atomtypes_ = atomtypes

        self.errors = errors # return errors

        if self.verbose > 0:
            elapsed = time.time() - t
            print('Done: [%dmin %dsec]'% (int(elapsed/60), int(elapsed%60)))
            sys.stdout.flush()    

    def predict(self, X, Xatoms, Xatomtypes, verbose=None, variance=False, print_kernel=False):
        '''
        Given some X values, return the predicted y values.
        '''
        t = time.time()

        if verbose is None:
            verbose = self.verbose
        
        y_ = np.empty(shape=(X.shape[0], len(self.lambs_)))
        
        if variance:
            pred_var = np.zeros((X.shape[0],))

        for i, gamma in enumerate(np.unique(self.gammas_)):
            for j, alpha in enumerate(np.unique(self.alphas_)):          

                if verbose > 0:
                    elapsed = time.time() - t
                    print("Combination %d of %d [%dmin %dsec]"% (i+1, len(np.unique(self.gammas_)),
                        int(elapsed/60), int(elapsed%60)))
                    sys.stdout.flush()
                
                y_list = [i for i in range(len(self.gammas_)) if self.gammas_[i] == gamma]
                kernel1 = self.global_kernel1(metric=self.kernel_func_, gamma=gamma, normalize_kernel=self.normalize_kernel)
                kernel2 = self.global_kernel2(degree=self.degree, normalize_kernel=self.normalize_kernel, alpha=self.rematch)
                K1 = kernel1.create(X, Xatoms, Xatomtypes, self.X_fit_, self.natoms_, self.atomtypes_)
                K2 = kernel2.create(X, Xatoms, Xatomtypes, self.X_fit_, self.natoms_, self.atomtypes_)
                K = (1-alpha)*K1 + alpha*K2
                y_[:, y_list] = np.dot(K, self.dual_coefs_[y_list].T)

            # if variance:
            #     K_test = kernel.create(X, Xatoms, Xatomtypes, X, Xatoms, Xatomtypes)
            #     V = solve_triangular(self.L_fit_[i], K.T, lower=True)
            #     v = np.sum(V*V, axis=0)
            #     pred_var = K_test.flat[::X.shape[0] + 1] - v
            #     pred_var = self.y_all_scaler.inverse_transform(pred_var)

        # PCA here if needed
        y = y_
        y = self.y_all_scaler.inverse_transform(y)
        if y.shape[1] == 1:
            y = y.flatten()
        
        if verbose > 0:
            elapsed = time.time() - t
            print('Done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()

        if variance:
            return y, pred_var
        elif print_kernel:
            return K
        else:
            return y

    def score(self, y_true, y_pred):
        scores = {}
        scores['rmse'] = np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
        scores['mae'] = np.round(mean_absolute_error(y_true, y_pred),2)
        scores['r2'] = np.round(r2_score(y_true, y_pred),2)
        return scores


class elemental_kernel_CV(BaseEstimator):
    def __init__(self, global_kernel='average', krr_param_grid=None, cv_nfolds=5, cv_shuffles=1, 
    verbose=True, copy_X=True, mae=False, print_errors=False, normalize_kernel=True):
        self.krr_param_grid = krr_param_grid # Grid of: gamma values, possible kernel funcs
        self.normalize_kernel = normalize_kernel
        if global_kernel == 'average':
            self.global_kernel = ElementalDimerAverageKernel
        elif global_kernel == 'sum':
            self.global_kernel = ElementalDimerSumKernel
        elif global_kernel == 'soap':
            self.global_kernel = ElementalSOAPKernel
        else:
            raise Exception('Unknown global kernel type.')
        self.cv_nfolds = cv_nfolds
        self.verbose = verbose
        self.cv_shuffles = cv_shuffles
        self.copy_X = copy_X
        self.mae = mae
        self.print_errors = print_errors

    def fit(self, X, y, natoms, atomtypes, show_plot=True, cv_indices=None, random_state=None):
        '''
        Fits the model using n-fold cross-validation.
            natoms (list): list of number of atoms in the first monomer (inefficient, I know)
            atomtypes (array of lists): element of each atom in the dimer
            cv_indices (array): option to provide own predefined CV splits.
            show_plot (bool): whether to show heatmap of CV values or not.
        '''
        t = time.time()

        if y.ndim < 2:
            y = y.reshape(-1,1)

        if cv_indices is None:
            cv_indices = np.arange(X.shape[0]) 

        kfold = RepeatedKFold(n_splits=self.cv_nfolds, n_repeats=self.cv_shuffles, random_state=random_state)
        cv_folds = kfold.split(X[cv_indices])
        n_cv_folds = kfold.get_n_splits()

        add_train_inds = np.setdiff1d(np.arange(X.shape[0]), cv_indices) # Empty if cv_indices was None
        cv_folds = list(cv_folds)
        cv_folds = [(np.concatenate((train_fold, add_train_inds)), test_fold) for train_fold, test_fold in cv_folds]
        # Returns list of (train folds, test folds) * n_cv

        if self.verbose > 0:
            elapsed = time.time() - t
            print('Starting cross-validation [%dmin %dsec]' % (int(elapsed/60), int(elapsed%60)))
            sys.stdout.flush()

        y_ = y
        errors = []

        for fold_i, (train_i, test_i) in enumerate(cv_folds):
            fold_errors = np.empty((len(self.krr_param_grid['gamma']),
                                    len(self.krr_param_grid['kernel_func']),
                                    len(self.krr_param_grid['lambda']),
                                    y_.shape[1]))
            if self.verbose > 0:
                elapsed = time.time() - t
                print('CV %d of %d [%dmin %dsec]' % (fold_i + 1,
                                                        n_cv_folds,
                                                        int(elapsed / 60),
                                                        int(elapsed % 60)))
                sys.stdout.flush()
            
            for gamma_i, gamma in enumerate(self.krr_param_grid['gamma']):
                if self.verbose > 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                
                for func_i, func in enumerate(self.krr_param_grid['kernel_func']):
                    if self.global_kernel == ElementalSOAPKernel:
                        kernel = self.global_kernel(degree=gamma, normalize_kernel=self.normalize_kernel)
                    else:
                        kernel = self.global_kernel(metric=func, gamma=gamma, normalize_kernel=self.normalize_kernel)
                    X_train, X_test = X[train_i], X[test_i]
                    train_atoms, test_atoms = natoms[train_i], natoms[test_i]
                    train_atomtypes, test_atomtypes = atomtypes[train_i], atomtypes[test_i]
                    K_train = kernel.create(X_train, train_atoms, train_atomtypes)
                    K_test = kernel.create(X_test, test_atoms, test_atomtypes, X_train, train_atoms, train_atomtypes)

                    for lamb_i, lamb in enumerate(self.krr_param_grid['lambda']):
                        y_train, y_test = y_[train_i], y_[test_i]
                        y_scaler = StandardScaler()
                        y_train = y_scaler.fit_transform(y_train)
                        y_test = y_scaler.transform(y_test)

                        if self.verbose > 0:
                            sys.stdout.write(',')
                            sys.stdout.flush()
                        K_train_ = K_train.copy()
                        K_train_.flat[::K_train_.shape[0] + 1] += lamb
                        try:
                            
                            L_ = cholesky(K_train_, lower=True)
                            x = solve_triangular(L_, y_train, lower=True)
                            dual_coef_ = solve_triangular(L_.T, x)
                            pred_mean = np.dot(K_test, dual_coef_)
                            pred_mean = y_scaler.inverse_transform(pred_mean)
                            y_test = y_scaler.inverse_transform(y_test)

                            if self.mae:
                                e = np.mean(np.abs(pred_mean - y_test), 0)
                                #e = np.mean(np.abs((pred_mean - y_test)/y_test), 0) # Percentage error
                            else: # return RMSE
                                e = np.sqrt(np.mean(((pred_mean - y_test) ** 2), 0))
                                #e = np.sqrt(np.mean(np.square(((y_test - pred_mean) / y_test)), axis=0)) # Percentage RMSE

                        except np.linalg.LinAlgError:
                            e = np.inf
                        fold_errors[gamma_i, func_i, lamb_i] = e
            
            if self.verbose > 0:
                sys.stdout.write('\n')
                sys.stdout.flush()
            errors.append(fold_errors)
        errors = np.array(errors)
        
        if self.print_errors:
                print('CV errors: ', errors.tolist())
                print('Average errors: ', np.mean(errors,0))
        errors = np.mean(errors, 0)  # average over folds

        self.dual_coefs_ = np.empty((y_.shape[1], X.shape[0]))
        self.lambs_ = np.empty(y_.shape[1])
        self.kernel_func_ = np.empty(y_.shape[1])
        self.gammas_ = np.empty(y_.shape[1])
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Refit [%dmin %dsec]' % (int(elapsed/60), int(elapsed%60)))
            sys.stdout.flush()
        # single_combo = True, means can automatically increase lambda (regularisation) value

        errors = np.mean(errors, -1)
        if self.verbose > 0:
            print('Gamma params: ', self.krr_param_grid['gamma'])
            print('Kernel functions: ', self.krr_param_grid['kernel_func'])
            print('Lambda params: ', self.krr_param_grid['lambda'])

        if show_plot:
            n_lambda = len(self.krr_param_grid['lambda'])
            n_gamma = len(self.krr_param_grid['gamma'])
            n_funcs = len(self.krr_param_grid['kernel_func'])
            if n_funcs <= 1:
                plt.imshow(errors.reshape(errors.shape[0], errors.shape[-1]), cmap='plasma')
                plt.xticks(range(n_lambda), map('{:.1e}'.format, list(self.krr_param_grid['lambda'])), rotation=45)
                plt.yticks(range(n_gamma), map('{:.1e}'.format, list(self.krr_param_grid['gamma'])))
                plt.xlabel('lambda')
                plt.ylabel('gamma')
                plt.colorbar()
                plt.show()
            else: # n_funcs > 1:
                fig, ax = plt.subplots(nrows=1, ncols=n_funcs, figsize=(10,4))
                for i in range(n_funcs):
                    data = errors[:, i]
                    im = ax[i].imshow(data, cmap='plasma')
                    ax[i].set_xlabel('lambda')
                    ax[i].set_ylabel('gamma')
                    ax[i].set_title('{} kernel'.format(self.krr_param_grid['kernel_func'][i]))

                    ax[i].set_xticks(range(n_lambda))
                    ax[i].set_xticklabels(list(self.krr_param_grid['lambda']), rotation=45)
                    ax[i].set_yticklabels(list(self.krr_param_grid['gamma']))
                    ax[i].set_yticks(range(n_gamma))
                    plt.colorbar(im, ax=ax[i])     
                plt.show()

        min_params = np.argsort(errors, axis=None)
        gamma_i, kernel_func_i, lambda_i = np.unravel_index(min_params[0], errors.shape)

        gamma = self.krr_param_grid['gamma'][gamma_i]
        kernel_func = self.krr_param_grid['kernel_func'][kernel_func_i]
        lamb = self.krr_param_grid['lambda'][lambda_i]

        print('CV results: gamma={}, lambda={}, kernel_func={}, RMSE={:.2f}'.format(gamma, 
                lamb, kernel_func, np.min(errors)))

        if len(self.krr_param_grid['gamma']) > 1:
            if gamma == self.krr_param_grid['gamma'][0]:
                print('Gamma at lower edge.')
            if gamma == self.krr_param_grid['gamma'][-1]:
                print('Gamma at upper edge.')
        if len(self.krr_param_grid['lambda']) > 1:
            if lamb == self.krr_param_grid['lambda'][0]:
                print('Lambda at lower edge.')
            if lamb == self.krr_param_grid['lambda'][-1]:
                print('Lambda at upper edge.')
        

        self.dual_coefs_ = np.empty((y_.shape[1], X.shape[0]))
        self.gammas_[:] = gamma
        self.kernel_func_ = kernel_func
        self.lambs_[:] = lamb

        combos = list(zip(self.gammas_, self.lambs_, repeat(self.kernel_func_)))
         # if more than one optimal set of gamma/lambda, but use only one kernel_func
        n_unique_combos = len(set(combos))
        self.L_fit_ = [None] * n_unique_combos
        for i, (gamma, lamb, func) in enumerate(set(combos)):
            if self.verbose > 0:
                elapsed = time.time() - t
                print('Parameter combinations ' +  '%d of %d [%dmin %dsec]' % (
                                            i+1, n_unique_combos,
                                            int(elapsed/60),int(elapsed%60)))
                sys.stdout.flush()
            y_list = [i for i in range(y_.shape[1]) if
                self.lambs_[i] == lamb and self.gammas_[i] == gamma and self.kernel_func_ == kernel_func]
            
            # Refit to the newly determined parameters 
            if self.global_kernel == ElementalSOAPKernel:
                self.final_kernel = self.global_kernel(degree=gamma, normalize_kernel=self.normalize_kernel)
            else:
                self.final_kernel = self.global_kernel(metric=kernel_func, gamma=gamma, normalize_kernel=self.normalize_kernel)
            K = self.final_kernel.create(X, natoms, atomtypes)
            self.y_all_scaler = StandardScaler()
            y_all = self.y_all_scaler.fit_transform(y_[:, y_list])

            while True: 
                K.flat[::K.shape[0] + 1] += lamb - (lamb/10)
                try:
                    if self.verbose > 0:
                        print('Trying Cholesky decomposition, lambda = ', lamb)
                    L_ = cholesky(K, lower=True)
                    self.L_fit_[i] = L_
                    x = solve_triangular(L_, y_all, lower=True)
                    dual_coef_ = solve_triangular(L_.T, x)
                    self.dual_coefs_[y_list] = dual_coef_.T.copy()
                    break
                except np.linalg.LinAlgError:
                    if self.verbose > 0:
                        print('Linalg error, increasing lambda.')
                    lamb *= 10
                    self.lambs_[0] = lamb

        if self.copy_X:
            self.X_fit_ = X.copy()
            self.natoms_ = natoms.copy()
            self.atomtypes_ = atomtypes.copy()
            self.y_fit_ = y.copy()
        else:
            self.X_fit_ = X
            self.y_fit_ = y
            self.natoms_ = natoms
            self.atomtypes_ = atomtypes

        self.errors = errors # return errors
        #self.kernel = K
        if self.verbose > 0:
            elapsed = time.time() - t
            print('Done: [%dmin %dsec]'% (int(elapsed/60), int(elapsed%60)))
            sys.stdout.flush()

    def predict(self, X, Xatoms, Xatomtypes, verbose=None, variance=False, print_kernel=False):
        '''
        Given some X values, return the predicted y values.
        '''
        t = time.time()

        if verbose is None:
            verbose = self.verbose
        
        y_ = np.empty(shape=(X.shape[0], len(self.lambs_)))
        
        if variance:
            pred_var = np.zeros((X.shape[0],))

        for i, gamma in enumerate(np.unique(self.gammas_)):

            if verbose > 0:
                elapsed = time.time() - t
                print("Combination %d of %d [%dmin %dsec]"% (i+1, len(np.unique(self.gammas_)),
                    int(elapsed/60), int(elapsed%60)))
                sys.stdout.flush()
            
            y_list = [i for i in range(len(self.gammas_)) if self.gammas_[i] == gamma]
            if self.global_kernel == ElementalSOAPKernel:
                kernel = self.global_kernel(degree=gamma, normalize_kernel=self.normalize_kernel)
            else:
                kernel = self.global_kernel(metric=self.kernel_func_, gamma=gamma, normalize_kernel=self.normalize_kernel)
            K = kernel.create(X, Xatoms, Xatomtypes, self.X_fit_, self.natoms_, self.atomtypes_)
            y_[:, y_list] = np.dot(K, self.dual_coefs_[y_list].T)

            if variance:
                K_test = kernel.create(X, Xatoms, Xatomtypes, X, Xatoms, Xatomtypes)
                V = solve_triangular(self.L_fit_[i], K.T, lower=True)
                v = np.sum(V*V, axis=0)
                pred_var = K_test.flat[::X.shape[0] + 1] - v
                pred_var = self.y_all_scaler.inverse_transform(pred_var)

        # PCA here if needed
        y = y_
        y = self.y_all_scaler.inverse_transform(y)

        if y.shape[1] == 1:
            y = y.flatten()
        
        if verbose > 0:
            elapsed = time.time() - t
            print('Done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
            sys.stdout.flush()
        
        if variance:
            return y, pred_var
        elif print_kernel:
            return K
        else:
            return y
    
    def score(self, y_true, y_pred):
        scores = {}
        scores['rmse'] = np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
        scores['mae'] = np.round(mean_absolute_error(y_true, y_pred),2)
        scores['r2'] = np.round(r2_score(y_true, y_pred),2)
        return scores


class hybrid_krr_model(BaseEstimator):
    def __init__(self, alpha, rematch, gamma, degree, lamb, metric, print_errors=True, normalize=True, verbose=False):
        self.alpha = alpha
        self.rematch = rematch
        self.gamma = gamma
        self.degree = degree
        self.lamb = lamb
        self.metric = metric
        self.normalize = normalize
        self.verbose = verbose
        self.kernel1 = ElementalDimerAverageKernel
        self.kernel2 = ElementalSOAPKernel
        self.print_errors = print_errors

    def fit(self, X_train, y_train, train_atoms, train_atomtypes):
        self.X_train = X_train
        self.train_atoms = train_atoms
        self.train_atomtypes = train_atomtypes
        t = time.time()
            
        self.final_kernel1 = self.kernel1(metric=self.metric, gamma=self.gamma, normalize_kernel=self.normalize)
        self.final_kernel2 = self.kernel2(degree=self.degree, normalize_kernel=self.normalize, alpha=self.rematch)        
            
        K1 = self.final_kernel1.create(X_train, train_atoms, train_atomtypes)
        K2 = self.final_kernel2.create(X_train, train_atoms, train_atomtypes)

        K = (1-self.alpha)*K1 + self.alpha*K2
        self.y_scaler = StandardScaler()
        y_train = y_train.reshape(-1,1)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        self.L_fit_ = [None]
                
        while True:
            K.flat[::K.shape[0] + 1] += self.lamb - (self.lamb/10)
            try:
                L_ = cholesky(K, lower=True)
                self.L_fit_[0] = L_
                x = solve_triangular(L_, y_train_scaled, lower=True)
                dual_coef_ = solve_triangular(L_.T, x)
                self.dual_coefs_ = dual_coef_.T.copy()
                break
            except np.linalg.LinAlgError:
                print('Linalg error, increasing lambda.')
                self.lamb *= 10
                print('Lambda: {}'.format(self.lamb))
        
        elapsed = time.time() - t
        if self.verbose:
            print('Fitting model done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))


    def predict(self, X_test, test_atoms, test_atomtypes):
        y_ = np.empty(shape=(X_test.shape[0],))
        t = time.time()
        K1 = self.final_kernel1.create(X_test, test_atoms, test_atomtypes, self.X_train, self.train_atoms,
                                    self.train_atomtypes)
        K2 = self.final_kernel2.create(X_test, test_atoms, test_atomtypes, self.X_train, self.train_atoms,
                                    self.train_atomtypes)

        K = (1-self.alpha)*K1 + self.alpha*K2
        y_ = np.dot(K, self.dual_coefs_.T)
        y = self.y_scaler.inverse_transform(y_)
        if y.shape[1] == 1:
            y = y.flatten()  

        elapsed = time.time() - t
        if self.verbose:
            print('Predictions done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))

        return y

# Dimer kernel model (use to train and test different pairwise metrics without cross-validation)
class elemental_krr_model(BaseEstimator):
    def __init__(self, gamma, lamb, metric, kernel='average', print_errors=True, normalize=True, verbose=False):
        self.gamma = gamma
        self.lamb = lamb
        self.metric = metric
        self.normalize = normalize
        self.verbose = verbose
        if kernel == 'average':
            self.kernel = ElementalDimerAverageKernel
        elif kernel == 'sum':
            self.kernel = ElementalDimerSumKernel
        elif kernel == 'soap':
            self.kernel = ElementalSOAPKernel
        else:
            raise Exception('Kernel not defined.')
        self.print_errors = print_errors
        
    def fit(self, X_train, y_train, train_atoms, train_atomtypes):
        self.X_train = X_train
        self.train_atoms = train_atoms
        self.train_atomtypes = train_atomtypes
        t = time.time()
        if self.kernel == ElementalSOAPKernel:
            self.final_kernel = self.kernel(degree=self.gamma, normalize_kernel=self.normalize)
        else:
            self.final_kernel = self.kernel(metric=self.metric, gamma=self.gamma, normalize_kernel=self.normalize)
            
        K = self.final_kernel.create(X_train, train_atoms, train_atomtypes)
        self.y_scaler = StandardScaler()
        y_train = y_train.reshape(-1,1)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        self.L_fit_ = [None]
                
        while True:
            K.flat[::K.shape[0] + 1] += self.lamb - (self.lamb/10)
            try:
                L_ = cholesky(K, lower=True)
                self.L_fit_[0] = L_
                x = solve_triangular(L_, y_train_scaled, lower=True)
                dual_coef_ = solve_triangular(L_.T, x)
                self.dual_coefs_ = dual_coef_.T.copy()
                break
            except np.linalg.LinAlgError:
                print('Linalg error, increasing lambda.')
                self.lamb *= 10
                print('Lambda: {}'.format(self.lamb))
        
        elapsed = time.time() - t
        if self.verbose:
            print('Fitting model done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))
            
    
    def predict(self, X_test, test_atoms, test_atomtypes, plot=False, variance=False):
        y_ = np.empty(shape=(X_test.shape[0],))
        t = time.time()
        K = self.final_kernel.create(X_test, test_atoms, test_atomtypes, self.X_train, self.train_atoms,
                                    self.train_atomtypes)
        y_ = np.dot(K, self.dual_coefs_.T)
                      
        y = y_
        y = self.y_scaler.inverse_transform(y)
        if y.shape[1] == 1:
            y = y.flatten()  
        
        if variance:
            K_test = self.final_kernel.create(X_test, test_atoms, test_atomtypes, X_test, test_atoms, test_atomtypes)
            V = solve_triangular(self.L_fit_[0], K.T, lower=True)
            v = np.sum(V*V, axis=0)
            pred_var = K_test.flat[::X_test.shape[0] + 1] - v
            pred_var = self.y_scaler.inverse_transform(pred_var)

        elapsed = time.time() - t
        if self.verbose:
            print('Predictions done [%dmin %dsec]' % (int(elapsed / 60), int(elapsed % 60)))

        if variance:
            return y, pred_var
        else:
            return y


# Get average metrics for random split over n runs

def average_nfold_errors(X, y, res, i_train, atoms, atomtypes, gamma, lamb, kernel='average', normalize=False, ntest=6, n=20, metric='polynomial'):
    rmse = []
    mae = []
    percent_rmse = []

    for i in tqdm(range(n)):
        i_test = []
        random_test_names = np.random.choice(res.loc[i_train][1], ntest, replace=False)
        
        for test_name in random_test_names:
            test_name = str(test_name.split('-d')[0]) + (str('-d'))
            i_test.extend([i for i in res.loc[res[1].str.contains(test_name)].index])
    
        i_test = np.array(i_test).ravel()
        i_train_smaller = np.setdiff1d(i_train, i_test)
        X_train, X_test = X[i_train_smaller], X[i_test]
        y_train, y_test = y[i_train_smaller], y[i_test]
        train_natoms, test_natoms = atoms[i_train_smaller], atoms[i_test]
        train_atomtypes, test_atomtypes = atomtypes[i_train_smaller], atomtypes[i_test]
    
        if i == 0:
            print("Training samples: ", len(i_train_smaller));print("Test samples: ", len(i_test))
   
        model = elemental_krr_model(kernel=kernel, normalize=normalize, verbose=False, gamma=gamma, lamb=lamb)
        model.fit(X_train, y_train, train_natoms, train_atomtypes, metric=metric)
        preds = model.predict(X_test, y_test, test_atoms=test_natoms, test_atomtypes=test_atomtypes, plot=False)
        d = get_metrics(y_test, preds)
        percent_rmse.append(d['%rmse'])
        mae.append(d['mae'])
        rmse.append(d['rmse'])
        
    print('Average MAE: ', np.round(np.mean(mae),2), '+/-', np.round(np.std(mae), 2))
    print('Average RMSE: ', np.round(np.mean(rmse),2), '+/-', np.round(np.std(rmse), 2))
    print('Average %RMSE: ', np.round(np.mean(percent_rmse),2), '+/-', np.round(np.std(percent_rmse), 2))
    
    return mae, rmse, percent_rmse
