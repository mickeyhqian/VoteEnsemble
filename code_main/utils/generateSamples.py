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
     elif kwargs['type'] == 'sym_pareto':
          # generate symmetric pareto distributed samples with mean the same as the original case
          paretoShapes = kwargs['params']
          for i in range(len(paretoShapes)):
               pos_sample = genPareto(n, rng, paretoShapes[i])
               neg_sample = -genPareto(n, rng, paretoShapes[i])
               arrays_list.append(pos_sample + neg_sample + paretoShapes[i]/(paretoShapes[i]-1))
     elif kwargs['type'] == 'sym_pareto_zeromean':
          paretoShapes = kwargs['params']
          for i in range(len(paretoShapes)):
               pos_sample = genPareto(n, rng, paretoShapes[i])
               neg_sample = -genPareto(n, rng, paretoShapes[i])
               arrays_list.append(pos_sample + neg_sample)
     elif kwargs['type'] == 'neg_pareto':
          # generate negative pareto distributed samples with mean the same as the original case
          paretoShapes = kwargs['params']
          for i in range(len(paretoShapes)):
               arrays_list.append(-genPareto(n, rng, paretoShapes[i])+2*paretoShapes[i]/(paretoShapes[i]-1))
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


def genSample_LASSO(n, rng, sample_args):
    # output: a numpy 2-d array, the first column is y, the rest are x
    # y = X*beta + sympareto noise
    X_sample = genSample_SSKP(n, rng, type = "pareto", params = sample_args["params"])
    beta = sample_args["beta_true"]
    noise_shape = sample_args["noise"]
    noise = genSample_SSKP(n, rng, type = "sym_pareto_zeromean", params = [noise_shape])
    y = np.dot(X_sample, np.reshape(beta, (-1,1))) + noise
    return np.hstack((y, X_sample))

def genSample_LR(n, rng, sample_args):
    # output: a numpy 2-d array, the first column is y, the rest are x
    # y = X*beta + sympareto noise
    X_sample = genSample_SSKP(n, rng, type = "pareto", params = sample_args["params"])
    X_noises = genSample_SSKP(n, rng, type = "sym_pareto", params = sample_args["params"])
    beta = sample_args["beta_true"]
    noise_shape = sample_args["noise"]
    noise = genSample_SSKP(n, rng, type = "sym_pareto_zeromean", params = [noise_shape])
    y = np.dot(X_sample, np.reshape(beta, (-1,1))) + noise
    return np.hstack((y, X_sample+0.1 * X_noises))