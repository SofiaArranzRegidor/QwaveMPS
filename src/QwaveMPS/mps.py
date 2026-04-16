import numpy as np


def left_norm_check(mps_bin):
    result = np.einsum('lir,lis->rs', np.conj(mps_bin), mps_bin)
    #return result
    return np.allclose(result, np.eye(mps_bin.shape[2]))

def right_norm_check(mps_bin):
    result = np.einsum('lir,mir->lm', mps_bin, np.conj(mps_bin))
    #return result
    return np.allclose(result, np.eye(mps_bin.shape[0]))