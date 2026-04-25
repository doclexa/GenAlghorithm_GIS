import numpy as np

from mkm_core import prepare_mkm_matrix_for_application

_glinprop = None

def init_worker(glinprop):
    global _glinprop
    _glinprop = glinprop

def calc_fitness_glin(Aglin_list):
    Aglin = np.array(Aglin_list).reshape(5, 5)
    invM = np.linalg.inv(prepare_mkm_matrix_for_application(Aglin))
    mkm = _glinprop @ invM

    fitness1 = np.sum(mkm < 0) / mkm.size
    fitness2 = (mkm[:,0] + mkm[:,1] < 0.3).sum() / len(mkm)

    return (0.7 * fitness1 + 0.3 * fitness2,)
