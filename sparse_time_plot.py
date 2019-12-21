import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg
import plotly.express as px
import plotly
from tqdm import tqdm
import time
from matplotlib import pyplot as plt


def t_comp(eps, delta):
    fig = 8./(eps**2) * np.log(2./delta)
    fig = int(fig+1)
    return fig

def c(p):
    fig = (2 + int(p))**(p+1)
    if p > 1:
        nafig = np.prod(np.array([abs((p - i + 1)/i) for i in range(1, int(p) + 1)]))
    else:
        nafig = 1
    return fig*nafig

def m_comp(c, p, eps, n):
    fig = 7* np.power(3*c*n/(p*eps), 1/p)
    return np.int(fig + 1)

def beta(m, c, p):
    fig = (c/p) * np.power(float(m)+1, -p)
    return fig

def hutch(X, p, m, t):
    y = []
    n = X.shape[0]
    for i in range(1,np.int(t)+1):
        g_i = np.random.binomial(1, 0.5, size=(n)) * 2 - 1
        v_k = np.squeeze(np.array(X@g_i))
        u_k = g_i.T@v_k
        a_k = p
        S_i_k = a_k*u_k
        for k in range(2, m+1):
            v_k = np.squeeze(np.array(X@v_k))
            u_k = g_i.T @v_k
            a_k = a_k* (p-(k-1))/k
            if np.abs(a_k) < 1e-8:
                break
            S_i_k = S_i_k + (((-1)**(k-1)) * a_k) * u_k
        y.append(S_i_k)
    y = np.array(y).mean()
    return y


def power_method(A, x0, num_iter): # 5 pts
    # enter your code here
    x = np.copy(x0)
    A_x = np.squeeze(np.array(A.dot(x)))
    l = x.T @ A_x
    for i in range(num_iter):
        x = np.squeeze(np.array(A.dot(x)))
        x = x / np.linalg.norm(x)
        A_x = np.squeeze(np.array(A.dot(x)))
        l = x.T @ A_x
    return x, l


def alpha(A, delta):
    n = A.shape[0]
    q = int(4.82 * np.log(1. / delta) + 1)
    t = int(0.5 * np.log(4 * n) + 1)
    max_lambda = 0
    for _ in range(q):
        x0 = np.random.binomial(1, 0.5, size=(n,)) * 2 - 1
        x, l = power_method(A, x0, t)
        if l > max_lambda:
            max_lambda = l
    return max_lambda


def vova_bravo_without(A, p, eps, delta):
    n = A.shape[0]
    t = t_comp(eps, delta)
    c_p = c(p)
    m = m_comp(c_p, p, eps, n)
    b_m = beta(m, c_p, p)
    a = alpha(A, delta)
    return np.power(a, p) * int((1+b_m)*n - hutch(np.eye(n) - A / a, p, m, t))


def compute_true_schatten(A, p):
    if not sparse.issparse(A):
        U, S, Vh = np.linalg.svd(A)
    else:
        U, S, Vh = sparse.linalg.svds(A, k=min(A.shape) - 1)
    return np.power(S, p).sum()


def calc_error(y_true, y_pred):
    return abs(y_true - y_pred) / y_true


def plot_size_dependence(schatten_true, schatten_iter, p=5, figsize=(20, 8)):
    dense_sizes = np.linspace(100, 10000, 7).astype(int)
    sparse_sizes = [200, 1000, 3000, 7000]

    fig, ax = plt.subplots(2, 1, figsize=(10, 15))
    err_dense = []
    err_sparse = []
    time_dense_true = []
    time_dense_approximate = []
    time_sparse_true = []
    time_sparse_approximate = []

    fig.suptitle(f'Dependence of {p}-Schatten norm error on matrix size', fontsize=16)

    for n_sparse in tqdm(sparse_sizes):
        #A = np.random.uniform(low=-10, high=10, size=(n_dense, n_dense))
        #A = A.T @ A
        #start = time.clock()
        #true = schatten_true(A, p=p)
        #time_dense_true.append(time.clock() - start)
        #start = time.clock()
        #approx = schatten_iter(A, p=p)
        #time_dense_approximate.append(time.clock() - start)
        #err_dense.append(calc_error(true, approx))

        A = scipy.sparse.random(n_sparse, n_sparse, density=0.1)
        A = A.T @ A
        start = time.clock()
        true = schatten_true(A, p=p)
        time_sparse_true.append(time.clock() - start)
        start = time.clock()
        approx = schatten_iter(A, p=p)
        time_sparse_approximate.append(time.clock() - start)
        err_sparse.append(calc_error(true, approx))

    #ax[0].set_title('For dense matrices')
    #ax[0].plot(dense_sizes, err_dense, label='error')
    #ax[0].grid()
    #ax[0].legend()
    #ax[0].set_xlabel('n')
    #ax[0].set_ylabel('error')
    ax[0].set_title('For sparse matrices')
    ax[0].plot(sparse_sizes, err_sparse, label='error')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('error')

    #ax[2].set_title('Time for dense matrices')
    #ax[2].plot(dense_sizes, time_dense_true, label='via svd')
    #ax[2].plot(dense_sizes, time_dense_approximate, label='approximate')
    #ax[2].grid()
    #ax[2].legend()
    #ax[2].set_xlabel('n')
    #ax[2].set_ylabel('time')

    ax[1].set_title('Time for sparse matrices')
    ax[1].plot(sparse_sizes, time_sparse_true, label='via svd')
    ax[1].plot(sparse_sizes, time_sparse_approximate, label='approximate')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('time')

    return fig, err_dense, err_sparse

schatten_iter = lambda A, p : vova_bravo_without(A, p, eps=0.1, delta=0.05)
fig, dense_res, sparse_res = plot_size_dependence(compute_true_schatten, schatten_iter, p=5)
fig.savefig('sparse_time.png')
