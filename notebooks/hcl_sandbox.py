# REF: https://heterocl.csl.cornell.edu/doc/tutorials/tutorial_01_get_started.html

import heterocl as hcl
import numpy as np

# initialize environment
hcl.init()

# define algorithm
def add_func(a,A,i,j):
    return A[i,j] + a
def simple_compute(a, A):
    B = hcl.compute(A.shape, lambda i,j: add_func(a,A,i,j), "B")
    # B = hcl.Tensor(A.shape)
    # with hcl.for_(0, A.shape[0], name="i") as i:
    #     with hcl.for_(0, A.shape[1], name="j") as j:
    #         B[i,j] = A[i,j] + a
    return B

# define inputs
a = hcl.placeholder((), "a")
A = hcl.placeholder((5,5), "A")

# apply hardware customization
sched = hcl.create_schedule([a,A], simple_compute)

# create executable
simple_comput_func = hcl.build(schedule=sched)

# Prepare inputs
hcl_a = 10
np_A = np.random.randint(100, size=A.shape)
hcl_A = hcl.asarray(np_A)
hcl_B = hcl.asarray(np.zeros(A.shape))

# run executable
simple_comput_func(hcl_a, hcl_A, hcl_B)

# print results
np_A = hcl_A.asnumpy()
np_B = hcl_B.asnumpy()
print("{}\n{}\n{}".format(hcl_a, np_A, np_B))