SCALE = 10

def unnormalize(x, lb, ub, scale=SCALE):
    return x / scale * (ub - lb) + lb

def normalize(x, lb, ub, scale=SCALE):
    return (x - lb) * scale / (ub - lb)