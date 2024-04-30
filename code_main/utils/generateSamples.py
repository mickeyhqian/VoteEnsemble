import numpy as np
from scipy import stats

# generate pareto distributed samples
def genPareto(n, rng, paretoShape):
    return stats.lomax.rvs(paretoShape, size=n, random_state=rng) + 1

def genTruncatedNormal(n, rng, mean, std):
     return np.maximum(stats.norm.rvs(loc=mean, scale=std, size=n, random_state=rng),0)

def genSample_SSKP(n, rng, **kwargs):
     arrays_list = []
     if kwargs['type'] == 'pareto':
          paretoShapes = kwargs['params']
          for i in range(len(paretoShapes)):
               arrays_list.append(genPareto(n, rng, paretoShapes[i]))
     elif kwargs['type'] == 'normal':
          mean, std = kwargs['params'][0], kwargs['params'][1]
          for i in range(len(mean)):
               arrays_list.append(genTruncatedNormal(n, rng, mean[i], std[i]))
     else:
          raise ValueError('Invalid type')
     return np.vstack(arrays_list).T 