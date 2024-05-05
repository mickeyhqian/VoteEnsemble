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


def genSample_portfolio(n, rng, **kwargs):
     # currently only consider uncorrelated samples
     if kwargs['type'] == 'pareto' or kwargs['type'] == 'normal':
          return genSample_SSKP(n, rng, type=kwargs['type'], params=kwargs['params'])
     elif kwargs['type'] == 'sym_pareto':
          # generate symmetric pareto distributed samples with mean specified by kwargs['params']
          paretoShapes = kwargs['params']
          pos_sample = genSample_SSKP(n, rng, type='pareto', params=paretoShapes)
          neg_sample = -genSample_SSKP(n, rng, type='pareto', params=paretoShapes)
          return pos_sample + neg_sample + np.array(paretoShapes)
     else:
          raise ValueError('Invalid type')

def genSample_network(n, rng, **kwargs):
     # simply assume that the samples are Pareto distributed
     s, c, g = kwargs['size']
     sample_S = genPareto((n,s,g), rng, kwargs['params'][0])
     sample_D = genPareto((n,c,g), rng, kwargs['params'][1])
     return np.concatenate((sample_S, sample_D), axis = 1)