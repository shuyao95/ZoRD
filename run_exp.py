import argparse
import pickle
import optax

from functions.synthetics import *
from functions.attacks import *
from functions.metrics import *

from optimizers import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
import copy

from jax import grad
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from utils import *
from bayes_opt import BayesianOptimization, UtilityFunction

def parse_arguments():
    parser = argparse.ArgumentParser(description='Synthetics experiments')
    
    # for synthetic functions
    parser.add_argument('--func', default='Ackley', type=str, help='type of synthetic functions')
    parser.add_argument('--dim', type=int, default=10, help='dimention of synthetic functions')
    parser.add_argument('--lb', type=float, default=-32, help='lower bound of the domain')
    parser.add_argument('--ub', type=float, default=32, help='upper bound of the domain')
    
    # for gradient descent (GD)-based algorithms
    parser.add_argument('--gd_opt', default='adam', type=str, help='type of gradient-based optimization, [sgd, adam, adgrad]')
    parser.add_argument('--gd_lr', default=0.1, type=float, help='learning rate ')
    parser.add_argument('--gd_iters', type=int, default=20, help='iterations for gradient-based opt')
    
    # for derivative estimation by finite difference (FD) method
    parser.add_argument('--q', default=10, type=int, help='for estimated gradient')
    parser.add_argument('--mu', default=0.01, type=float, help='for estimated gradient')
    
    # for gradientless descent (GLD) algorithm
    parser.add_argument('--max_r', default=5, type=int, help='for estimated gradient')
    parser.add_argument('--min_r', default=0.1, type=float, help='for estimated gradient')
    parser.add_argument('--cond', default=10, type=float, help='for estimated gradient')
    
    # others
    parser.add_argument('--n_inits', type=int, default=15, help='number of initial points')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--alg', type=str, default="zord", help='zo optimization algorithms')
    parser.add_argument('--save', type=str, default="./data/standard-", help='filename to save')
    parser.add_argument('--load', type=str, default="", help='filename to load for gradient-based optimization')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    args = parser.parse_args()
    
    return args
    
if __name__ == '__main__':
    args = parse_arguments()
    rng = jrandom.PRNGKey(args.seed)
    np.random.seed(args.seed)
    
    data = None
    # loading from queries from BO or hypercube as initializations
    if len(args.load) > 0:
        print("loding from", args.load)
        with open(args.load, 'rb') as file:
            data = pickle.load(file)
        
    multi_queries = []
    for i in range(args.n_runs):
        f = eval(args.func)(dim=args.dim, lb=args.lb, ub=args.ub)
        f = jit(f) if args.func in ['Ackley', 'Levy'] else f
            
        # initialization
        inits = jnp.asarray(np.random.uniform(size=(args.n_inits, args.dim), low=0, high=SCALE))
        # inits = jnp.vstack([inits, SCALE / 2 * jnp.ones((1, args.dim))]) # only for non-differentiable metric optimization
        init_queries = [(x, f(x)) for x in inits]
        min_idx = jnp.argmin(jnp.array([x[-1] for x in init_queries]))
        gd_init = inits[min_idx.item()]
            
        # for bo optimizer
        pbounds = {}
        for j in range(args.dim):
            pbounds['x%d'%j] = (0, SCALE)
        bo_optimizer = BayesianOptimization(f=None, pbounds=pbounds, random_state=args.seed + i)
        bo_utility = UtilityFunction(kind="ucb", kappa=2, xi=0)
        for q in init_queries:
            bo_optimizer.register(params=q[0], target=-q[-1])
        
        # construct GD optimizer
        params = {'learning_rate': args.gd_lr}
        gd_optimizer = eval("optax." + args.gd_opt)(**params)
        
        if data:
            # otain intializations from other sources
            queries = copy.deepcopy(data[i][:args.bo_iters])
            min_idx = jnp.argmin(jnp.array([q[-1] for q in queries]))
            gd_init = queries[min_idx.item()][0]
        else:
            # otain intializations from random search
            queries = init_queries
        
        # choose which zo optimization algorithm to use
        if args.alg == 'zord':
            gd_opt_params = [gd_optimizer, args.gd_iters]
            queries = zord_opt(f, gd_opt_params, bo_optimizer._gp, queries=queries)
        elif args.alg == 'gp_ucb':
            opt_params = [bo_optimizer, bo_utility, args.bo_iters]
            queries = gp_ucb_opt(f, opt_params)
        elif args.alg == 'turbo1':
            opt_params = [args.n_inits, args.bo_iters]
            queries = turbo1_opt(f, opt_params)
        elif args.alg == 'turbom':
            opt_params = [args.n_inits, args.bo_iters, args.regions]
            queries = turbom_opt(f, opt_params)
        elif args.alg == 'gd':
            opt_params = [gd_optimizer, args.gd_iters]
            queries = gd_opt(f, gd_init, opt_params)
        elif args.alg == 'rgf':
            opt_params = [gd_optimizer, args.gd_iters, args.q, args.mu]
            queries = rgf_opt(f, gd_init, opt_params)
        elif args.alg == 'es':
            opt_params = [gd_optimizer, args.gd_iters, args.q, args.mu]
            queries = es_opt(f, gd_init, opt_params)
        elif args.alg == 'prgf':
            opt_params = [gd_optimizer, args.gd_iters, args.q, args.mu]
            queries = prgf_opt(f, gd_init, opt_params)
        elif args.alg == 'gld':
            opt_params = [args.max_r, args.min_r, args.gd_iters]
            queries = gld_opt(f, gd_init, opt_params)
        elif args.alg == 'gld-fast':
            opt_params = [args.max_r, args.cond, args.gd_iters]
            queries = gld_fast_opt(f, gd_init, opt_params)
        else:
            raise ValueError('Do not support zo optimization algorithm: %s' %args.alg)
        
        print(len(queries), np.min([q[-1] for q in queries]))