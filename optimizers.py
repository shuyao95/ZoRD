import jax.numpy as jnp
from jax import grad, jit
import optax
import turbo

import GPy
import math
import numpy as np
import warnings
from copy import deepcopy
from scipy.optimize import approx_fprime

from utils import SCALE


# uncerinaty approximation using SE/RBF kernel
def grad_uncertainty(X, x, obs_noise=0.01):
    x = x.reshape(1,-1)
    dim = x.shape[-1]
    
    kernel = GPy.kern.RBF(input_dim=dim, lengthscale=1)
    var = kernel.dK2_dXdX2(x, x, 0, 0) - kernel.dK2_dXdX2(x, X, 0, 0) @ np.linalg.inv(kernel.K(X, X) + obs_noise * np.identity(X.shape[0])) @ kernel.dK2_dXdX2(X, x, 0, 0)
    return var.item()

def select_query(queries, greedy=False):
    xs = [q[0] for q in queries]
    ys = [np.asarray(q[1]).item() for q in queries]
    
    if greedy:
        ind = np.argmin(ys)
    else:
        ind = -1
    return xs[ind], ys[ind]


def post_mean_grad(gp, x):
    f = lambda x: gp.predict(x.reshape(1, -1)).item()
    grads = approx_fprime(np.asarray(x), f, epsilon=1e-20)
    return jnp.asarray(grads)


def rand_grad_est(f, x, q=1, mu=0.01):
    dim = len(x)
    samples = np.random.normal(0, 1, size=(q, dim))
    # orthogonalization
    orthos = []
    for u in samples:
        for ou in orthos:
            u = u - np.vdot(u, ou) * ou
        u = u / np.linalg.norm(u, ord=2)
        orthos.append(u)
        
    fx, grads = f(x), 0
    
    local_queries = []
    for s in orthos:
        s = jnp.asarray(s)
        new_x = x + mu * s
        new_fx = f(new_x)
        grads += ((new_fx - fx) / mu) * s
        local_queries += [(new_x, new_fx)]
        
    grads /= len(samples)
    return grads, local_queries


def rand_variation(f, x, q=1, mu=0.01):
    dim = len(x)
    samples = np.random.normal(0, 1, size=(q, dim))
    # orthogonalization
    orthos = []
    for u in samples:
        for ou in orthos:
            u = u - np.vdot(u, ou) * ou
        u = u / np.linalg.norm(u, ord=2)
        orthos.append(u)
        
    fx, variation = f(x), 0
    
    local_queries = []
    for s in orthos:
        s = jnp.asarray(s)
        new_x = x + mu * s
        new_fx = f(new_x)
        variation += (new_fx / mu) * s
        local_queries += [(new_x, new_fx)]
        
    variation /= q
    return variation, local_queries


def rand_grad_est_with_prior(f, x, priors, q=1, mu=0.01):
    dim = len(x)
    samples = np.random.normal(0, 1, size=(q, dim))
    samples = np.vstack([samples, priors])
    # orthogonalization
    orthos = []
    for u in samples:
        for ou in orthos:
            u = u - np.vdot(u, ou) * ou
        u = u / np.linalg.norm(u, ord=2)
        orthos.append(u)
        
    fx, grads = f(x), 0
    
    local_queries = []
    for s in orthos:
        s = jnp.asarray(s)
        new_x = x + mu * s
        new_fx = f(new_x)
        grads += ((new_fx - fx) / mu) * s
        local_queries += [(new_x, new_fx)]
        
    grads /= len(samples)
    return grads, local_queries
    

def gp_fit(gp, queries, max_queries=150, target_x=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        xs = np.array([q[0].tolist() for q in queries])
        ys = np.array([np.asarray(q[1]).item() for q in queries])
        dim = xs[0].shape[-1]
        if max_queries > 0: # use local queries to estimate the gradient more accurately
            if target_x is None:
                target_x = xs[np.argmin(ys).item()]
            dists = np.array([np.linalg.norm(x - target_x) for x in xs])
            idx = np.argsort(dists)[:max_queries]
            idx = sorted(idx)
            xs, ys = xs[idx], ys[idx]
        gp.fit(xs, ys)
        return xs

def zord_opt(f,gd_opt_params, gp, queries=[]):
    gd_optimizer, gd_iters = gd_opt_params
            
    # gd optimize 
    non_imp_iters, k = 0, 0
    min_fx = None
    tolerance = 10
    gd_state = None
    
    next_point, value = select_query(queries, greedy=True)
    gd_state = gd_optimizer.init(next_point)
    
    for k in range(gd_iters):
        next_point, value = select_query(queries, greedy=True)
            
        # re-init optimizer because of the inaccurate derivative estimation
        if min_fx is None or min_fx > value:
            min_fx = value
            non_imp_iters = 0
        else:
            non_imp_iters += 1
        if non_imp_iters > tolerance: 
            gd_state = gd_optimizer.init(next_point)
            non_imp_iters = 0
            
        steps = 0
        while True:
            xs = gp_fit(gp, queries, target_x=next_point)
            grads = post_mean_grad(gp, next_point)
            grads_uncertainty = grad_uncertainty(xs, next_point)
            update = (grads_uncertainty < 0.35) or (steps == 0)
            if update and steps < 10: # for reliable GD updates
                updates, gd_state = gd_optimizer.update(grads, gd_state) 
                next_point = optax.apply_updates(next_point, updates)
                steps += 1
            else:
                break
        
        next_point = jnp.clip(next_point, a_min=0, a_max=SCALE)
        value = f(next_point)
        queries += [(next_point, value)]
        print("Queries #%03d, value %.6f" %(len(queries), value))
        
    return queries

def gp_ucb_opt(f, opt_params):
    bo_optimizer, bo_utility, bo_iters = opt_params
    
    queries = []
    # bo update
    for t in range(bo_iters):
        next_point = jnp.array(list(bo_optimizer.suggest(bo_utility).values()))
        next_value = f(next_point)
        bo_optimizer.register(params=next_point, target=-next_value)
        queries += [(next_point, next_value)]
        print("Queries #%03d, value %.6f" %(len(queries), next_value))
    return queries


def turbo1_opt(f, opt_params):
    n_inits, n_iters = opt_params
    queries = []
    dim = len(f.lb)
    # bo update
    opt = turbo.Turbo1(
    # opt = Turbo1(
        f=lambda x: f(jnp.asarray(x)),  # Handle to objective function
        lb=np.zeros(dim),  # Numpy array specifying lower bounds
        ub=SCALE * np.ones(dim),  # Numpy array specifying upper bounds
        n_init=n_inits,  # Number of initial bounds from an Latin hypercube design
        max_evals = n_iters + n_inits,  # Maximum number of evaluations
        batch_size=1,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=4000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    
    opt.optimize()
    queries = [(x, y) for x, y in zip(opt.X, opt.fX)]
    return queries


def turbom_opt(f, opt_params):
    n_inits, n_iters, n_regions = opt_params
    queries = []
    dim = len(f.lb)
    # bo update
    opt = turbo.TurboM(
        f=lambda x: f(jnp.asarray(x)),  # Handle to objective function
        lb=np.zeros(dim),  # Numpy array specifying lower bounds
        ub=SCALE * np.ones(dim),  # Numpy array specifying upper bounds
        n_init=n_iters + n_inits,  # Number of initial bounds from an Latin hypercube design
        max_evals = n_iters,  # Maximum number of evaluations
        batch_size=1,  # How large batch size TuRBO uses
        n_trust_regions=n_regions, # Number of trust regions
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    
    opt.optimize()
    queries = [(x, y) for x, y in zip(opt.X, opt.fX)]
    return queries


def gd_opt(f, init_point, opt_params):
    gd_optimizer, gd_iters = opt_params
    grad_func = jit(grad(f))
    
    next_point = init_point
    gd_state = gd_optimizer.init(next_point)
    
    init_query = (init_point, f(init_point))
    queries = [init_query]
    # gd optimize 
    for k in range(gd_iters):
        next_point, value = select_query(queries, greedy=True)
        grads = grad_func(next_point)
        updates, gd_state = gd_optimizer.update(grads, gd_state)
        next_point = optax.apply_updates(next_point, updates)
        # projected into the domains
        next_point = jnp.clip(next_point, a_min=0, a_max=SCALE)
        value = f(next_point)
        queries += [(next_point, value)]
        print("Queries #%03d, value %.6f" %(len(queries), value))
    return queries


def rgf_opt(f, init_point, opt_params):
    gd_optimizer, gd_iters, q, mu = opt_params
    
    gd_iters = gd_iters // (q+1) + 1
    
    next_point = init_point
    gd_state = gd_optimizer.init(next_point)
    
    init_query = (init_point, f(init_point))
    queries = [init_query]
    # gd optimize 
    for k in range(gd_iters):
        next_point, value = select_query(queries, greedy=True)
        grads, local_queries = rand_grad_est(f, next_point, q=q, mu=mu)
        queries += local_queries
        updates, gd_state = gd_optimizer.update(grads, gd_state)
        next_point = optax.apply_updates(next_point, updates)
        # projected into the domains
        next_point = jnp.clip(next_point, a_min=0, a_max=SCALE)
        value = f(next_point)
        
        queries += [(next_point, value)]
        print("Queries #%03d, value %.6f" %(len(queries), value))
    return queries


def es_opt(f, init_point, opt_params):
    gd_optimizer, gd_iters, q, mu = opt_params
    
    gd_iters = gd_iters // (q+1) + 1
    
    next_point = init_point
    gd_state = gd_optimizer.init(next_point)
    
    init_query = (init_point, f(init_point))
    queries = [init_query]
    # gd optimize 
    for k in range(gd_iters):
        next_point, value = select_query(queries, greedy=True)
        variation, local_queries = rand_variation(f, next_point, q=q, mu=mu)
        queries += local_queries
        updates, gd_state = gd_optimizer.update(variation, gd_state)
        next_point = optax.apply_updates(next_point, updates)
        # projected into the domains
        next_point = jnp.clip(next_point, a_min=0, a_max=SCALE)
        value = f(next_point)
        
        queries += [(next_point, value)]
        print("Queries #%03d, value %.6f" %(len(queries), value))
    return queries


def prgf_opt(f, init_point, opt_params):
    gd_optimizer, gd_iters, q, mu = opt_params
    
    gd_iters = gd_iters // (q+2) + 1
    
    next_point = init_point
    gd_state = gd_optimizer.init(next_point)
    
    init_query = (init_point, f(init_point))
    queries = [init_query]
    prior = np.random.normal(0, 1, size=(1, len(init_point)))
    
    for k in range(gd_iters):
        next_point, value = select_query(queries, greedy=True)

        current_point = deepcopy(next_point)
        grads, local_queries = rand_grad_est_with_prior(f, next_point, prior, q=q, mu=mu)
        queries += local_queries
        updates, gd_state = gd_optimizer.update(grads, gd_state)
        next_point = optax.apply_updates(next_point, updates)
        # projected into the domains
        next_point = jnp.clip(next_point, a_min=0, a_max=SCALE)
        value = f(next_point)
        
        prior = current_point - next_point
        queries += [(next_point, value)]
        print("Queries #%03d, value %.6f" %(len(queries), value))
    return queries


def gld_opt(f, init_point, gld_opt_params):
    max_r, min_r, gld_iters = gld_opt_params
    K = int(math.log2(max_r / min_r)) 
    gld_iters = int(gld_iters / K) + 1
    
    next_point = init_point
    dim = len(next_point)
    
    init_query = (init_point, f(init_point))
    queries = [init_query]
    
    for k in range(gld_iters):
        for s in range(K+1):
            r = 0.5 ** s * max_r
            v = r * np.random.normal(0, 1, size=(1, dim)).reshape(-1)
            candidate = next_point + jnp.asarray(v)
            candidate = jnp.clip(candidate, a_min=0, a_max=SCALE)
            value = f(candidate)
            queries += [(candidate, value)]
        
        idx = jnp.argmin(jnp.array([q[-1] for q in queries])).item()
        next_point, value = queries[idx][0], queries[idx][1]
        print("Queries #%03d, value %.6f" %(len(queries), value))
    return queries


def gld_fast_opt(f, init_point, gld_opt_params):
    max_r, cond, gld_iters = gld_opt_params
    dim = len(init_point)
    K = int(math.log2(4 * math.sqrt(cond))) 
    H = int(dim * cond * math.log2(cond))
    gld_iters = int(gld_iters / (2 *K + 1)) + 1
    
    next_point = init_point
    
    init_query = (init_point, f(init_point))
    queries = [init_query]
    # gd optimize 
    for k in range(gld_iters):
        if (k + 1) % H == 0:
            max_r /= 2
        for s in range(2*K+1):
            r = 0.5 ** (s - K) * max_r
            v = r * np.random.normal(0, 1, size=(1, dim)).reshape(-1)
            candidate = next_point + jnp.asarray(v)
            candidate = jnp.clip(candidate, a_min=0, a_max=SCALE)
            value = f(candidate)
            queries += [(candidate, value)]
        
        idx = jnp.argmin(jnp.array([q[-1] for q in queries])).item()
        next_point, value = queries[idx][0], queries[idx][1]
        print("Queries #%03d, value %.6f" %(len(queries), value))
    return queries
