import multiprocessing
import numpy as np
from functools import partial

def rand_error(val):
    if val > 0.5:
        raise ValueError
    else:
        return val+1

if __name__ == "__main__":

    pool = multiprocessing.Pool(4)

    n_vals = 64
    vals = list(np.random.rand(n_vals))

    func = partial(rand_error)
    func_iter = pool.imap(func, vals)

    results = []
    for i in range(n_vals):
        try:
            results.append(func_iter.next())
        except ValueError:
            results.append(0)
    
    print(results)


