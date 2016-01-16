# cython: boundscheck=False, wraparound=False

cimport cython
cimport numpy as np
from numpy.math cimport INFINITY, expl, isinf, logl

import numpy as np

cdef inline double _logsumexp(double[:] X) nogil:
    # Builtin 'max' is unrolled for speed.
    cdef double X_max = -INFINITY
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]

    if isinf(X_max):
        return -INFINITY

    cdef double acc = 0
    for i in range(X.shape[0]):
        acc += expl(X[i] - X_max)

    return logl(acc) + X_max


def _forward(int n_samples, int n_components,
        double[:] log_startprob,
        double[:, :] log_transmat,
        double[:, :] framelogprob,
        double[:, :] fwdlattice):

    cdef int t, i, j
    cdef double[::1] work_buffer = np.zeros(n_components)

    with nogil:
        for i in range(n_components):
            fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]

        for t in range(1, n_samples):
            for j in range(n_components):
                for i in range(n_components):
                    work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]

                fwdlattice[t, j] = _logsumexp(work_buffer) + framelogprob[t, j]


def _backward(int n_samples, int n_components,
        double[::1] log_startprob,
        double[:, :] log_transmat,
        double[:, :] framelogprob,
        double[:, :] bwdlattice):

    cdef int t, i, j
    cdef double logprob
    cdef double[::1] work_buffer = np.zeros(n_components)

    with nogil:
        for i in range(n_components):
            bwdlattice[n_samples - 1, i] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[i, j]
                                      + framelogprob[t + 1, j]
                                      + bwdlattice[t + 1, j])
                bwdlattice[t, i] = _logsumexp(work_buffer)


def _compute_lneta(int n_samples, int n_components,
        double[:, :] fwdlattice,
        double[:, :] log_transmat,
        double[:, :] bwdlattice,
        double[:, :] framelogprob,
        double[:, :, :] lneta):

    cdef double logprob = _logsumexp(fwdlattice[n_samples - 1])
    cdef int t, i, j

    with nogil:
        for t in range(n_samples - 1):
            for i in range(n_components):
                for j in range(n_components):
                    lneta[t, i, j] = (fwdlattice[t, i]
                                      + log_transmat[i, j]
                                      + framelogprob[t + 1, j]
                                      + bwdlattice[t + 1, j]
                                      - logprob)


def _viterbi(int n_samples, int n_components,
        double[:] log_startprob, # n_components
        double[:, :] log_transmat,  # n_components x n_components
        double[:, :] framelogprob): # n_samples x n_components

    cdef int c0, c1, t, max_pos
    cdef double[:, ::1] viterbi_lattice
    cdef int[::1] state_sequence
    cdef double logprob
    cdef double[:, ::1] work_buffer

    # Initialization
    state_sequence_arr = np.empty(n_samples, dtype=np.int32)
    state_sequence = state_sequence_arr
    viterbi_lattice = np.zeros((n_samples, n_components))
    work_buffer = np.empty((n_components, n_components))

    # viterbi_lattice[0] = log_startprob + framelogprob[0]
    for c1 in range(n_components):
        viterbi_lattice[0, c1] = log_startprob[c1] + framelogprob[0, c1]

    # Induction
    #for t in range(1, n_samples):
    #    work_buffer = viterbi_lattice[t-1] + log_transmat.T
    #    viterbi_lattice[t] = np.max(work_buffer, axis=1) + framelogprob[t]
    cdef double buf, maxbuf
    for t in range(1, n_samples):
        for c0 in range(n_components):
            maxbuf = -INFINITY
            for c1 in range(n_components):
                buf = log_transmat[c1, c0] + viterbi_lattice[t-1, c1]
                work_buffer[c0, c1] = buf
                if buf > maxbuf:
                    maxbuf = buf

            viterbi_lattice[t, c0] = maxbuf + framelogprob[t, c0]

    # Observation traceback
    #max_pos = np.argmax(viterbi_lattice[n_samples - 1, :])
    maxbuf = -INFINITY
    for c1 in range(n_components):
        buf = viterbi_lattice[n_samples - 1, c1]
        if buf > maxbuf:
            maxbuf = buf
            max_pos = c1

    state_sequence[n_samples - 1] = max_pos
    logprob = viterbi_lattice[n_samples - 1, max_pos]

    for t in range(n_samples - 2, -1, -1):
        maxbuf = -INFINITY
        for c1 in range(n_components):
            tmp = viterbi_lattice[t, c1] + log_transmat[c1, state_sequence[t + 1]]
            if tmp > maxbuf:
                maxbuf = tmp
                max_pos = c1
        state_sequence[t] = max_pos

    return state_sequence_arr, logprob
