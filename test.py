from BaseLearners import BaseNetwork
import numpy as np
from numpy.typing import NDArray
from multiprocessing.pool import Pool
from scipy import stats



def evaluateObjective(network: BaseNetwork, learningResult: NDArray, sample: NDArray) -> NDArray:
    return network.objective(learningResult, sample)


if __name__ == "__main__":
    # s, p, c, g = 3, 2, 3, 5
    # # C = np.array([0.84902017, 0.73141691])
    # C = np.array([0.95, 1.25])
    # # Q_sp = np.array([[[0.33959485, 0.93963167, 0.11099611, 0.29228903, 0.32671781],
    # #                     [0.11183712, 0.68568047, 0.89458674, 0.6290198 , 0.45712153]],
    # #                     [[0.30562386, 0.25141794, 0.52145227, 0.93541722, 0.09209385],
    # #                     [0.32229624, 0.78332703, 0.97653641, 0.18445241, 0.03020424]],
    # #                     [[0.3414426 , 0.45228102, 0.22978903, 0.05338668, 0.24009525],
    # #                     [0.92622445, 0.08852084, 0.84160819, 0.02508719, 0.46203217]]])
    # Q_sp = np.ones((s, p, g)) * 0.99
    # # Q_pc = np.array([[[0.08967017, 0.08001494, 0.29854161, 0.6979279 , 0.12176636],
    # #                     [0.96381038, 0.1531089 , 0.38302768, 0.61996695, 0.90823239],
    # #                     [0.70189534, 0.39932218, 0.55522219, 0.00271483, 0.89190262]],
    # #                     [[0.00499716, 0.94445126, 0.88466098, 0.5649098 , 0.54066034],
    # #                     [0.35775422, 0.14473548, 0.17228491, 0.64621086, 0.6801251 ],
    # #                     [0.22298374, 0.17262673, 0.42564852, 0.25968014, 0.01474239]]])
    # Q_pc = np.ones((p, c, g)) * 0.99
    # # Q_pc = np.ones((p, c, g)) * 0.99
    # R = np.array([[0.75762425, 0.60499893, 0.64820431, 0.86142707, 0.01262116],
    #                 [0.67859456, 0.37513025, 0.86190007, 0.18231789, 0.20731563]])
    # # M = np.array([0.96766421, 0.5369463])
    # M = np.array([0.99, 0.99])
    # # M = np.array([2, 1])
    # # H = np.array([[0.19395032, 0.81790575, 0.78516185, 0.20974165, 0.90638053],
    # #                 [0.00704014, 0.96558659, 0.15616022, 0.7461778 , 0.73652027],
    # #                 [0.72572877, 0.98864562, 0.39330404, 0.68259888, 0.92669604]])
    # H = np.ones((c, g)) * 1.25
    # # H = rngProb.uniform(low = 1, high = 1.3, size = (c, g))

    # # paretoParams =  [1.90394177, 2.00408536]
    # paretoParams = [2.1, 2.1]

    # def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
    #     sampleS: NDArray = stats.lomax.rvs(paretoParams[0], size = (n, s, g), random_state = rng) + 1
    #     sampleD: NDArray = stats.lomax.rvs(paretoParams[1], size = (n, c, g), random_state = rng) + 1
    #     sampleQsp: NDArray = stats.lomax.rvs((Q_sp + 1) / Q_sp, size = (n, s, p, g), random_state = rng)
    #     sampleQpc: NDArray = stats.lomax.rvs((Q_pc + 1) / Q_pc, size = (n, p, c, g), random_state = rng)
    #     sampleM: NDArray = stats.lomax.rvs((M + 1) / M, size = (n, p), random_state = rng)
    #     return np.concatenate((sampleQsp.reshape(n, -1), sampleQpc.reshape(n, -1), sampleS.reshape(n, -1), sampleD.reshape(n, -1), sampleM), axis = 1)
    
    # s, p, c, g = 3, 2, 3, 5
    # # C = np.array([0.84902017, 0.73141691])
    # C = np.array([0.9, 1.25])
    # # Q_sp = np.array([[[0.33959485, 0.93963167, 0.11099611, 0.29228903, 0.32671781],
    # #                     [0.11183712, 0.68568047, 0.89458674, 0.6290198 , 0.45712153]],
    # #                     [[0.30562386, 0.25141794, 0.52145227, 0.93541722, 0.09209385],
    # #                     [0.32229624, 0.78332703, 0.97653641, 0.18445241, 0.03020424]],
    # #                     [[0.3414426 , 0.45228102, 0.22978903, 0.05338668, 0.24009525],
    # #                     [0.92622445, 0.08852084, 0.84160819, 0.02508719, 0.46203217]]])
    # Q_sp = np.ones((s, p, g))
    # # Q_pc = np.array([[[0.08967017, 0.08001494, 0.29854161, 0.6979279 , 0.12176636],
    # #                     [0.96381038, 0.1531089 , 0.38302768, 0.61996695, 0.90823239],
    # #                     [0.70189534, 0.39932218, 0.55522219, 0.00271483, 0.89190262]],
    # #                     [[0.00499716, 0.94445126, 0.88466098, 0.5649098 , 0.54066034],
    # #                     [0.35775422, 0.14473548, 0.17228491, 0.64621086, 0.6801251 ],
    # #                     [0.22298374, 0.17262673, 0.42564852, 0.25968014, 0.01474239]]])
    # Q_pc = np.ones((p, c, g))
    # # Q_pc = np.ones((p, c, g)) * 0.99
    # R = np.array([[0.75762425, 0.60499893, 0.64820431, 0.86142707, 0.01262116],
    #                 [0.67859456, 0.37513025, 0.86190007, 0.18231789, 0.20731563]])
    # # M = np.array([0.96766421, 0.5369463])
    # M = np.array([1,1])
    # # M = np.array([2, 1])
    # # H = np.array([[0.19395032, 0.81790575, 0.78516185, 0.20974165, 0.90638053],
    # #                 [0.00704014, 0.96558659, 0.15616022, 0.7461778 , 0.73652027],
    # #                 [0.72572877, 0.98864562, 0.39330404, 0.68259888, 0.92669604]])
    # H = np.ones((c, g)) * 1.25
    # # H = rngProb.uniform(low = 1, high = 1.3, size = (c, g))

    # # paretoParams =  [1.90394177, 2.00408536]
    # paretoParams = [2.1, 2.1]

    # def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
    #     sampleS: NDArray = stats.lomax.rvs(paretoParams[0], size = (n, s, g), random_state = rng) + 1
    #     sampleD: NDArray = stats.lomax.rvs(paretoParams[1], size = (n, c, g), random_state = rng) + 1
    #     sampleQsp: NDArray = stats.lomax.rvs((100 * Q_sp + 1) / (100 * Q_sp), size = (n, s, p, g), random_state = rng) / 100
    #     sampleQpc: NDArray = stats.lomax.rvs((100 * Q_pc + 1) / (100 * Q_pc), size = (n, p, c, g), random_state = rng) / 100
    #     sampleM: NDArray = stats.lomax.rvs((100 * M + 1) / (100 * M), size = (n, p), random_state = rng) / 100
    #     return np.concatenate((sampleQsp.reshape(n, -1), sampleQpc.reshape(n, -1), sampleS.reshape(n, -1), sampleD.reshape(n, -1), sampleM), axis = 1)

    # rngEval = np.random.default_rng(seed = 777)

    # network = BaseNetwork3(s, C, R, H)

    # s, p, c, g = 3, 2, 3, 5
    # C = np.array([1, 50])
    # # Q_sp = np.array([[[0.33959485, 0.93963167, 0.11099611, 0.29228903, 0.32671781],
    # #                     [0.11183712, 0.68568047, 0.89458674, 0.6290198 , 0.45712153]],
    # #                     [[0.30562386, 0.25141794, 0.52145227, 0.93541722, 0.09209385],
    # #                     [0.32229624, 0.78332703, 0.97653641, 0.18445241, 0.03020424]],
    # #                     [[0.3414426 , 0.45228102, 0.22978903, 0.05338668, 0.24009525],
    # #                     [0.92622445, 0.08852084, 0.84160819, 0.02508719, 0.46203217]]])
    # Q_sp = np.concatenate([np.ones((s, 1, g)), 1e-2 * np.ones((s, 1, g))], axis=1)
    # # Q_sp = rngProb.uniform(low = 0.99, high = 1.0, size = (s, p, g))
    # # Q_pc = np.array([[[0.08967017, 0.08001494, 0.29854161, 0.6979279 , 0.12176636],
    # #                     [0.96381038, 0.1531089 , 0.38302768, 0.61996695, 0.90823239],
    # #                     [0.70189534, 0.39932218, 0.55522219, 0.00271483, 0.89190262]],
    # #                     [[0.00499716, 0.94445126, 0.88466098, 0.5649098 , 0.54066034],
    # #                     [0.35775422, 0.14473548, 0.17228491, 0.64621086, 0.6801251 ],
    # #                     [0.22298374, 0.17262673, 0.42564852, 0.25968014, 0.01474239]]])
    # # Q_pc = np.ones((p, c, g))
    # Q_pc = np.concatenate([np.ones((1, c, g)), 1e-2 * np.ones((1, c, g))], axis=0)
    # # Q_pc = rngProb.uniform(low = 0.99, high = 1.0, size = (p, c, g))
    # # Q_pc = np.ones((p, c, g)) * 0.99
    # R = np.array([[0.75762425, 0.60499893, 0.64820431, 0.86142707, 0.01262116],
    #                 [0.67859456, 0.37513025, 0.86190007, 0.18231789, 0.20731563]])
    # # M = np.array([0.96766421, 0.5369463])
    # M = np.array([1000, 1000])
    # # M = np.array([2, 1])
    # # H = np.array([[0.19395032, 0.81790575, 0.78516185, 0.20974165, 0.90638053],
    # #                 [0.00704014, 0.96558659, 0.15616022, 0.7461778 , 0.73652027],
    # #                 [0.72572877, 0.98864562, 0.39330404, 0.68259888, 0.92669604]])
    # H = np.ones((c, g)) * 5
    # # H = rngProb.uniform(low = 3, high = 5, size = (c, g))

    # # paretoParams =  [1.90394177, 2.00408536]
    # # paretoParams =  [2, 2.01]
    # # paretoParams =  [3, 1.2]
    # paramS = np.ones((s, g)) * 5
    # paramD = np.ones((c, g)) * 5
    # paramD[0, 0] = 1.1
    
    s, p, c, g = 3, 2, 3, 5
    C = np.array([1.5, 40])
    # Q_sp = np.array([[[0.33959485, 0.93963167, 0.11099611, 0.29228903, 0.32671781],
    #                     [0.11183712, 0.68568047, 0.89458674, 0.6290198 , 0.45712153]],
    #                     [[0.30562386, 0.25141794, 0.52145227, 0.93541722, 0.09209385],
    #                     [0.32229624, 0.78332703, 0.97653641, 0.18445241, 0.03020424]],
    #                     [[0.3414426 , 0.45228102, 0.22978903, 0.05338668, 0.24009525],
    #                     [0.92622445, 0.08852084, 0.84160819, 0.02508719, 0.46203217]]])
    Q_sp = np.concatenate([np.ones((s, 1, g)), 1e-2 * np.ones((s, 1, g))], axis=1)
    # Q_sp = rngProb.uniform(low = 0.99, high = 1.0, size = (s, p, g))
    # Q_pc = np.array([[[0.08967017, 0.08001494, 0.29854161, 0.6979279 , 0.12176636],
    #                     [0.96381038, 0.1531089 , 0.38302768, 0.61996695, 0.90823239],
    #                     [0.70189534, 0.39932218, 0.55522219, 0.00271483, 0.89190262]],
    #                     [[0.00499716, 0.94445126, 0.88466098, 0.5649098 , 0.54066034],
    #                     [0.35775422, 0.14473548, 0.17228491, 0.64621086, 0.6801251 ],
    #                     [0.22298374, 0.17262673, 0.42564852, 0.25968014, 0.01474239]]])
    # Q_pc = np.ones((p, c, g))
    Q_pc = np.concatenate([np.ones((1, c, g)), 1e-2 * np.ones((1, c, g))], axis=0)
    # Q_pc = rngProb.uniform(low = 0.99, high = 1.0, size = (p, c, g))
    # Q_pc = np.ones((p, c, g)) * 0.99
    R = np.array([[0.75762425, 0.60499893, 0.64820431, 0.86142707, 0.01262116],
                    [0.67859456, 0.37513025, 0.86190007, 0.18231789, 0.20731563]])
    # M = np.array([0.96766421, 0.5369463])
    M = np.array([1000, 1000])
    # M = np.array([2, 1])
    # H = np.array([[0.19395032, 0.81790575, 0.78516185, 0.20974165, 0.90638053],
    #                 [0.00704014, 0.96558659, 0.15616022, 0.7461778 , 0.73652027],
    #                 [0.72572877, 0.98864562, 0.39330404, 0.68259888, 0.92669604]])
    H = np.ones((c, g)) * 5
    # H = rngProb.uniform(low = 3, high = 5, size = (c, g))

    # paretoParams =  [1.90394177, 2.00408536]
    # paretoParams =  [2, 2.01]
    # paretoParams =  [3, 1.2]
    paramS = np.ones((s, g)) * 5
    paramD = np.ones((c, g)) * 5
    paramD[0, 0] = 2.1

    def sampler(n: int, repIdx: int, rng: np.random.Generator) -> NDArray:
        # sampleS = stats.lomax.rvs(paretoParams[0], size = (n, s, g), random_state = rng) + 1
        # sampleD = stats.lomax.rvs(paretoParams[1], size = (n, c, g), random_state = rng) + 1
        sampleS = stats.lomax.rvs(paramS, size = (n, s, g), random_state = rng) + 1000
        sampleD = stats.lomax.rvs(paramD, size = (n, c, g), random_state = rng) + 1
        return np.concatenate((sampleS, sampleD), axis = 1)

    rngEval = np.random.default_rng(seed = 777)
    network = BaseNetwork(C, Q_sp, Q_pc, R, M, H)
    numParallel = 14
    
    def evaluator(learningResult: NDArray, evalSample: NDArray) -> float:
        interval = max(1, len(evalSample) // numParallel + 1)
        with Pool(numParallel) as pool:
            results = pool.starmap(
                evaluateObjective,
                [(network, learningResult, evalSample[i:min(len(evalSample), i + interval)]) for i in range(0, len(evalSample), interval)],
                chunksize = 1
            )
        return np.mean(np.concatenate(results))

    sampleSize = 1000
    numReplicates = 20
    solCount = {}
    for i in range(numReplicates):
        newSample = sampler(sampleSize, -1, rngEval)
        solution = tuple(network.learn(newSample))
        if solution not in solCount:
            solCount[solution] = 0
        solCount[solution] += 1
        
    for sol, count in solCount.items():
        print(f"Solution {sol} count = {count} frequency = {count / numReplicates}")
    
    # x0 = np.array([0,0])
    # x1 = np.array([1,0])
    # x2 = np.array([0,1])
    # x3 = np.array([1,1])
    # values = []
    
    
    # evalSample = sampler(100000, -1, rngEval)
    # values.append([])
    # values[-1].append(evaluator(x0, evalSample))
    # values[-1].append(evaluator(x1, evalSample))
    # values[-1].append(evaluator(x2, evalSample))
    # values[-1].append(evaluator(x3, evalSample))
    
    # evalSample = sampler(100000, -1, rngEval)
    # values.append([])
    # values[-1].append(evaluator(x0, evalSample))
    # values[-1].append(evaluator(x1, evalSample))
    # values[-1].append(evaluator(x2, evalSample))
    # values[-1].append(evaluator(x3, evalSample))
    
    # evalSample = sampler(100000, -1, rngEval)
    # values.append([])
    # values[-1].append(evaluator(x0, evalSample))
    # values[-1].append(evaluator(x1, evalSample))
    # values[-1].append(evaluator(x2, evalSample))
    # values[-1].append(evaluator(x3, evalSample))
    
    # for i in range(len(values)):
    #     print(f"values for sample {i}")
    #     print("[0,0] = ", np.mean(values[i][0]))
    #     print("[1,0] = ", np.mean(values[i][1]))
    #     print("[0,1] = ", np.mean(values[i][2]))
    #     print("[1,1] = ", np.mean(values[i][3]))