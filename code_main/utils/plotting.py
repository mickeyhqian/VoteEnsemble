import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def plot_final(SAA_obj_avg, bagging_obj_avg, sample_number, B_list, k_list):
    _, ax = plt.subplots()
    ax.plot(sample_number, SAA_obj_avg, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            ax.plot(sample_number, bagging_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'B={B_list[ind1]}, k={k_list[ind2]}')
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective', size = 20)
    ax.legend()
    fig_name  = "obj_avg_" + str(B_list) + '_' +str(k_list) + ".png"
    plt.savefig(fig_name)
    # plt.show()
    return

def plot_CI_final(SAA_obj_list, bagging_obj_list, sample_number, B_list, k_list):
    number_of_iterations = len(SAA_obj_list[0])
    _, ax = plt.subplots()
    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            diff_Bk = []
            for i in range(len(sample_number)):
                diff_Bk.append([bagging_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
            
            diff_Bk_mean = np.mean(diff_Bk, axis = 1)
            diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
            diff_conf_int = 1.96 * diff_Bk_std_err

            ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'B={B}, k={k}')
            ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)
    
    ax.axhline(0, color='grey', linewidth=2, linestyle='--') 
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective difference', size = 20)
    ax.legend()
    fig_name  = "obj_CI_" + str(B_list) + '_' + str(k_list) + ".png"
    plt.savefig(fig_name)
    # plt.show()
    return

def plot_optGap_twoPhase(obj_opt, minmax, SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B_list, k_list, B12_list, name = None):
    # minmax: specify whether the problem is minimization/maximization
    _, ax = plt.subplots()
    SAA_obj_gap = [obj_opt - obj for obj in SAA_obj_avg] if minmax == "max" else [obj - obj_opt for obj in SAA_obj_avg]
    ax.plot(sample_number, SAA_obj_gap, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    
    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            bagging_obj_gap = [obj_opt - obj for obj in bagging_alg1_obj_avg[ind1][ind2]] if minmax == "max" else [obj - obj_opt for obj in bagging_alg1_obj_avg[ind1][ind2]]
            ax.plot(sample_number, bagging_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1, B={B_list[ind1]}, k={k_list[ind2]}')
    
    for ind1, B12 in enumerate(B12_list):
        for ind2, k in enumerate(k_list):
            if B12 == "X" or k == "X":
                continue
            bagging_obj_gap = [obj_opt - obj for obj in bagging_alg3_obj_avg[ind1][ind2]] if minmax == "max" else [obj - obj_opt for obj in bagging_alg3_obj_avg[ind1][ind2]]
            ax.plot(sample_number, bagging_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12_list[ind1]}, k={k_list[ind2]}')
            
            bagging_obj_gap = [obj_opt - obj for obj in bagging_alg4_obj_avg[ind1][ind2]] if minmax == "max" else [obj - obj_opt for obj in bagging_alg4_obj_avg[ind1][ind2]]
            ax.plot(sample_number, bagging_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12_list[ind1]}, k={k_list[ind2]}')
    
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Optimality Gap', size = 20)
    ax.legend(fontsize = 'small')
    if name == None:
        fig_name = "opt_gap_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + ".png"
    else:
        fig_name = name + "_opt_gap" + ".png"
    plt.savefig(fig_name)
    return

def plot_optTime_twoPhase(obj_opt, SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, sample_number, B_list, k_list, B12_list):
    _, ax = plt.subplots()
    number_of_iterations = len(SAA_obj_list[0])
    # count number of times the optimal objective is achieved
    SAA_opt_times  = [ sum([1 for obj in obj_list if abs(obj - obj_opt) < 1e-6])/number_of_iterations for obj_list in SAA_obj_list ]
    ax.plot(sample_number, SAA_opt_times, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            bagging_opt_times = [sum([1 for obj in obj_list if abs(obj - obj_opt) < 1e-6])/number_of_iterations for obj_list in bagging_alg1_obj_list[ind1][ind2]]
            ax.plot(sample_number, bagging_opt_times, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1, B={B_list[ind1]}, k={k_list[ind2]}')
    
    for ind1, B12 in enumerate(B12_list):
        for ind2, k in enumerate(k_list):
            if B12 == "X" or k == "X":
                continue
            bagging_opt_times = [sum([1 for obj in obj_list if abs(obj - obj_opt) < 1e-6])/number_of_iterations for obj_list in bagging_alg3_obj_list[ind1][ind2]]
            ax.plot(sample_number, bagging_opt_times, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12_list[ind1]}, k={k_list[ind2]}')
            bagging_opt_times = [sum([1 for obj in obj_list if abs(obj - obj_opt) < 1e-6])/number_of_iterations for obj_list in bagging_alg4_obj_list[ind1][ind2]]
            ax.plot(sample_number, bagging_opt_times, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12_list[ind1]}, k={k_list[ind2]}')
    
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Optimal Achieved Ratio', size = 20)
    ax.legend(fontsize = 'small')
    fig_name = "opt_time_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + ".png"
    plt.savefig(fig_name)
    return


def plot_optGap_epsilonComparison(obj_opt, minmax, SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B, k, B12, epsilon_list):
    # similar plotting function, but used to compare different epsilon values
    _, ax = plt.subplots()
    SAA_obj_gap = [obj_opt - obj for obj in SAA_obj_avg] if minmax == "max" else [obj - obj_opt for obj in SAA_obj_avg]
    ax.plot(sample_number, SAA_obj_gap, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    alg1_obj_gap = [obj_opt - obj for obj in bagging_alg1_obj_avg] if minmax == "max" else [obj - obj_opt for obj in bagging_alg1_obj_avg]
    ax.plot(sample_number, alg1_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1')

    for ind, epsilon in enumerate(epsilon_list):
        if epsilon == "X":
            continue
        alg3_obj_gap = [obj_opt - obj for obj in bagging_alg3_obj_avg[ind]] if minmax == "max" else [obj - obj_opt for obj in bagging_alg3_obj_avg[ind]]
        ax.plot(sample_number, alg3_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, eps={epsilon}')
        alg4_obj_gap = [obj_opt - obj for obj in bagging_alg4_obj_avg[ind]] if minmax == "max" else [obj - obj_opt for obj in bagging_alg4_obj_avg[ind]]
        ax.plot(sample_number, alg4_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, eps={epsilon}')
    
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Optimality Gap', size = 20)
    ax.legend(fontsize = 'small')
    ax.set_title(f'B={B}, k={k}, B12={B12}')
    fig_name = "opt_gap_" + str(B) + '_' +str(k) + '_' + str(B12) + '_' + str(epsilon_list) +".png"
    plt.savefig(fig_name)
    return

def plot_probComparison(dist_1, dist_2, sample_number, name):
    n_groups = len(sample_number)
    bar_width = 0.35  # Width of the bars
    index = np.arange(n_groups)  # Position of groups

    fig, ax = plt.subplots()

    ax.bar(index - bar_width/2, dist_1, bar_width, label='Setting 1', color='b', edgecolor='black')
    ax.bar(index + bar_width/2, dist_2, bar_width, label='Setting 2', color='r', edgecolor='black')

    ax.set_xlabel('Sample Number')
    if name == 'prob_opt':
        ax.set_ylabel('Probability')
    elif name == 'prob_diff':
        ax.set_ylabel('Probability Difference')
    ax.set_title(name)
    ax.set_xticks(index)
    ax.set_xticklabels(sample_number)
    ax.legend()
    # plt.show()
    plt.savefig(name + '.png')
    
    return


def plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B_list, k_list, B12_list, name = None, skip_alg1 = False):
    _, ax = plt.subplots()
    ax.plot(sample_number, SAA_obj_avg, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    if not skip_alg1:
        for ind1, B in enumerate(B_list):
            for ind2, k in enumerate(k_list):
                if B == "X" or k == "X":
                    continue
                ax.plot(sample_number, bagging_alg1_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1, B={B_list[ind1]}, k={k_list[ind2]}')
    
    for ind1, B12 in enumerate(B12_list):
        for ind2, k in enumerate(k_list):
            if B12 == "X" or k == "X":
                continue
            ax.plot(sample_number, bagging_alg3_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12_list[ind1]}, k={k_list[ind2]}')
            ax.plot(sample_number, bagging_alg4_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12_list[ind1]}, k={k_list[ind2]}')
    
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective', size = 20)
    ax.legend(fontsize = 'small')
    if name == None:
        fig_name = "obj_avg_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + ".png"
    else:
        fig_name = name + "_obj_avg" + ".png"
    plt.savefig(fig_name)
    return

def plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, sample_number, B_list, k_list, B12_list, name = None, skip_alg1 = False):
    number_of_iterations = len(SAA_obj_list[0])
    _, ax = plt.subplots()
    if not skip_alg1:
        for ind1, B in enumerate(B_list):
            for ind2, k in enumerate(k_list):
                if B == "X" or k == "X":
                    continue
                diff_Bk = []
                for i in range(len(sample_number)):
                    diff_Bk.append([bagging_alg1_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
                
                diff_Bk_mean = np.mean(diff_Bk, axis = 1)
                diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
                diff_conf_int = 1.96 * diff_Bk_std_err

                ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1, B={B}, k={k}')
                ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)

    for ind1, B12 in enumerate(B12_list):
        for ind2, k in enumerate(k_list):
            if B12 == "X" or k == "X":
                continue
            diff_Bk = []
            for i in range(len(sample_number)):
                diff_Bk.append([bagging_alg3_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
            
            diff_Bk_mean = np.mean(diff_Bk, axis = 1)
            diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
            diff_conf_int = 1.96 * diff_Bk_std_err

            ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12}, k={k}')
            ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)
            
            diff_Bk = []
            for i in range(len(sample_number)):
                diff_Bk.append([bagging_alg4_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
            
            diff_Bk_mean = np.mean(diff_Bk, axis = 1)
            diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
            diff_conf_int = 1.96 * diff_Bk_std_err

            ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12}, k={k}')
            ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)
    
    ax.axhline(0, color='grey', linewidth=2, linestyle='--') 
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective Difference', size = 20)
    ax.legend(fontsize = 'small')
    if name == None:
        fig_name = "obj_CI_" + str(B_list) + '_' + str(k_list) + '_' + str(B12_list) + ".png"
    else:
        fig_name = name + "_obj_CI" + ".png"
    plt.savefig(fig_name)
    return

def plot_epsilonComparison(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B, k, B12, epsilon_list):
    # similar plotting function, but used to compare different epsilon values
    _, ax = plt.subplots()
    if SAA_obj_avg is not None:
        ax.plot(sample_number, SAA_obj_avg, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    if bagging_alg1_obj_avg is not None:
        ax.plot(sample_number, bagging_alg1_obj_avg, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1')
    
    if bagging_alg3_obj_avg is not None:
        for ind, epsilon in enumerate(epsilon_list):
            if epsilon == "X":
                continue
            ax.plot(sample_number, bagging_alg3_obj_avg[ind], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, eps={epsilon}')
    if bagging_alg4_obj_avg is not None:    
        for ind, epsilon in enumerate(epsilon_list):
            if epsilon == "X":
                continue
            ax.plot(sample_number, bagging_alg4_obj_avg[ind], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, eps={epsilon}')
    
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective', size = 20)
    ax.set_title(f'B={B}, k={k}, B12={B12}')
    ax.legend(fontsize = 'small')
    fig_name = "obj_avg_" + str(B) + '_' +str(k) + '_' + str(B12) + '_' + str(epsilon_list) +".png"
    plt.savefig(fig_name)
    return

def plot_CI_epsilonComparison(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, sample_number, B, k, B12, epsilon_list):
    # similar plotting function, but used to compare different epsilon values
    number_of_iterations = len(SAA_obj_list[0])
    _, ax = plt.subplots()
    
    diff_Bk = []
    for i in range(len(sample_number)):
        diff_Bk.append([bagging_alg1_obj_list[i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
    diff_Bk_mean = np.mean(diff_Bk, axis = 1)
    diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
    diff_conf_int = 1.96 * diff_Bk_std_err
    ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1')
    ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)

    for ind, epsilon in enumerate(epsilon_list):
        if epsilon == "X":
            continue
        diff_Bk = []
        for i in range(len(sample_number)):
            diff_Bk.append([bagging_alg3_obj_list[ind][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
        diff_Bk_mean = np.mean(diff_Bk, axis = 1)
        diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
        diff_conf_int = 1.96 * diff_Bk_std_err
        ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, eps={epsilon}')
        ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)

        diff_Bk = []
        for i in range(len(sample_number)):
            diff_Bk.append([bagging_alg4_obj_list[ind][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
        diff_Bk_mean = np.mean(diff_Bk, axis = 1)
        diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
        diff_conf_int = 1.96 * diff_Bk_std_err
        ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, eps={epsilon}')
        ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)

    ax.axhline(0, color='grey', linewidth=2, linestyle='--')
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective Difference', size = 20)
    ax.legend(fontsize = 'small')
    ax.set_title(f'B={B}, k={k}, B12={B12}')
    fig_name = "obj_diff_" + str(B) + '_' +str(k) + '_' + str(B12) + '_' + str(epsilon_list) +".png"
    plt.savefig(fig_name)
    return


def plot_params(SAA_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B12_list, k_list, name):
    _, ax = plt.subplots()
    for ind1, B12 in enumerate(B12_list):
        for ind2, k in enumerate(k_list):
            if B12 == "X" or k == "X":
                continue
            ax.plot(sample_number, [bagging_alg3_obj_avg[ind1][ind2][i] - SAA_obj_avg[i] for i in range(len(sample_number))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12_list[ind1]}, k={k_list[ind2]}')
            ax.plot(sample_number, [bagging_alg4_obj_avg[ind1][ind2][i] - SAA_obj_avg[i] for i in range(len(sample_number))], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12_list[ind1]}, k={k_list[ind2]}')
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective difference', size = 20)
    ax.legend(fontsize = 'small')
    fig_name = name + ".png"
    plt.savefig(fig_name)
    return

def plot_droComparison(SAA_obj_avg, dro_wasserstein_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B_list, k_list, B12_list, varepsilon_list, name = None):
    _, ax = plt.subplots()
    ax.plot(sample_number, SAA_obj_avg, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    for ind, varepsilon in enumerate(varepsilon_list):
        if varepsilon == "X":
            continue
        ax.plot(sample_number, dro_wasserstein_obj_avg[ind], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'DRO, $\epsilon$={varepsilon}')
    
    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            ax.plot(sample_number, bagging_alg1_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1, B={B_list[ind1]}, k={k_list[ind2]}')
    
    for ind1, B12 in enumerate(B12_list):
        for ind2, k in enumerate(k_list):
            if B12 == "X" or k == "X":
                continue
            ax.plot(sample_number, bagging_alg3_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12_list[ind1]}, k={k_list[ind2]}')
            ax.plot(sample_number, bagging_alg4_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12_list[ind1]}, k={k_list[ind2]}')
    
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective', size = 20)
    ax.legend(fontsize = 'small')
    if name == None:
        fig_name = "obj_avg_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + '_' + str(varepsilon_list)+ ".png"
    else:
        fig_name = name + ".png"
    plt.savefig(fig_name)
    return

def plot_CI_droComparison(SAA_obj_list, dro_wasserstein_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, sample_number, B_list, k_list, B12_list, varepsilon_list):
    number_of_iterations = len(SAA_obj_list[0])
    _, ax = plt.subplots()
    
    for ind, varepsilon in enumerate(varepsilon_list):
        if varepsilon == "X":
            continue
            
        diff_varepsilon = []
        for i in range(len(sample_number)):
            diff_varepsilon.append([dro_wasserstein_obj_list[ind][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
        
        diff_varepsilon_mean = np.mean(diff_varepsilon, axis = 1)
        diff_varepsilon_std_err = stats.sem(diff_varepsilon, axis = 1)
        diff_conf_int = 1.96 * diff_varepsilon_std_err

        ax.plot(sample_number, diff_varepsilon_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'DRO, $\epsilon$={varepsilon}')
        ax.fill_between(sample_number, diff_varepsilon_mean - diff_conf_int, diff_varepsilon_mean + diff_conf_int, alpha = 0.2)

    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            diff_Bk = []
            for i in range(len(sample_number)):
                diff_Bk.append([bagging_alg1_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
            
            diff_Bk_mean = np.mean(diff_Bk, axis = 1)
            diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
            diff_conf_int = 1.96 * diff_Bk_std_err

            ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1, B={B}, k={k}')
            ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)

    for ind1, B12 in enumerate(B12_list):
        for ind2, k in enumerate(k_list):
            if B12 == "X" or k == "X":
                continue
            diff_Bk = []
            for i in range(len(sample_number)):
                diff_Bk.append([bagging_alg3_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
            
            diff_Bk_mean = np.mean(diff_Bk, axis = 1)
            diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
            diff_conf_int = 1.96 * diff_Bk_std_err

            ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12}, k={k}')
            ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)
            
            diff_Bk = []
            for i in range(len(sample_number)):
                diff_Bk.append([bagging_alg4_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
            
            diff_Bk_mean = np.mean(diff_Bk, axis = 1)
            diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
            diff_conf_int = 1.96 * diff_Bk_std_err

            ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12}, k={k}')
            ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)
    
    ax.axhline(0, color='grey', linewidth=2, linestyle='--') 
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective Difference', size = 20)
    ax.legend(fontsize = 'small')
    fig_name = "obj_diff_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + "_" + str(varepsilon_list) + ".png"
    plt.savefig(fig_name)
    return

def plot_optGap_droComparison(obj_opt, minmax, SAA_obj_avg, dro_wasserstein_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B_list, k_list, B12_list, varepsilon_list):
    # minmax: specify whether the problem is minimization/maximization
    _, ax = plt.subplots()
    SAA_obj_gap = [obj_opt - obj for obj in SAA_obj_avg] if minmax == "max" else [obj - obj_opt for obj in SAA_obj_avg]
    ax.plot(sample_number, SAA_obj_gap, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    
    for ind, varepsilon in enumerate(varepsilon_list):
        if varepsilon == "X":
            continue
        dro_obj_gap = [obj_opt - obj for obj in dro_wasserstein_obj_avg[ind]] if minmax == "max" else [obj - obj_opt for obj in dro_wasserstein_obj_avg[ind]]
        ax.plot(sample_number, dro_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'DRO, $\epsilon$={varepsilon}')
    
    for ind1, B in enumerate(B_list):
        for ind2, k in enumerate(k_list):
            if B == "X" or k == "X":
                continue
            bagging_obj_gap = [obj_opt - obj for obj in bagging_alg1_obj_avg[ind1][ind2]] if minmax == "max" else [obj - obj_opt for obj in bagging_alg1_obj_avg[ind1][ind2]]
            ax.plot(sample_number, bagging_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1, B={B_list[ind1]}, k={k_list[ind2]}')
    
    for ind1, B12 in enumerate(B12_list):
        for ind2, k in enumerate(k_list):
            if B12 == "X" or k == "X":
                continue
            bagging_obj_gap = [obj_opt - obj for obj in bagging_alg3_obj_avg[ind1][ind2]] if minmax == "max" else [obj - obj_opt for obj in bagging_alg3_obj_avg[ind1][ind2]]
            ax.plot(sample_number, bagging_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12_list[ind1]}, k={k_list[ind2]}')
            
            bagging_obj_gap = [obj_opt - obj for obj in bagging_alg4_obj_avg[ind1][ind2]] if minmax == "max" else [obj - obj_opt for obj in bagging_alg4_obj_avg[ind1][ind2]]
            ax.plot(sample_number, bagging_obj_gap, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12_list[ind1]}, k={k_list[ind2]}')
    
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Optimality Gap', size = 20)
    ax.legend(fontsize = 'small')
    fig_name = "opt_gap_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + '_' + str(varepsilon_list) + ".png"
    plt.savefig(fig_name)
    return


def plot_many_methods(obj_avg_dicts, sample_number):
    _, ax = plt.subplots()
    fig_name = "obj_avg_"
    for key in obj_avg_dicts:
        ax.plot(sample_number, obj_avg_dicts[key], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = key)
        fig_name += key + "_"
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective', size = 20)
    ax.legend(fontsize = 'small')
    plt.savefig(fig_name)
    return

def plot_CI_many_methods(obj_list_dicts, sample_number):
    number_of_iterations = len(obj_list_dicts[list(obj_list_dicts.keys())[0]][0])
    _, ax = plt.subplots()
    fig_name = "obj_diff_"
    for key in obj_list_dicts:
        if key == "SAA":
            continue
        diff_key = []
        for i in range(len(sample_number)):
            diff_key.append([obj_list_dicts[key][i][j] - obj_list_dicts["SAA"][i][j] for j in range(number_of_iterations)])
        
        diff_key_mean = np.mean(diff_key, axis = 1)
        diff_key_std_err = stats.sem(diff_key, axis = 1)
        diff_conf_int = 1.96 * diff_key_std_err

        ax.plot(sample_number, diff_key_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = key)
        ax.fill_between(sample_number, diff_key_mean - diff_conf_int, diff_key_mean + diff_conf_int, alpha = 0.2)
        fig_name += key + "_"
    
    ax.axhline(0, color='grey', linewidth=2, linestyle='--')
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective Difference', size = 20)
    ax.legend(fontsize = 'small')
    plt.savefig(fig_name)
    return


def plot_test(SAA_obj_avg, bagging_alg3_obj_avg, bagging_alg3_trick_obj_avg, sample_number, k_list, B1_list, name = None):
    _, ax = plt.subplots()
    ax.plot(sample_number, SAA_obj_avg, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
    for ind1, B1 in enumerate(B1_list):
        for ind2, k in enumerate(k_list):
            if B1 == "X" or k == "X":
                continue
            ax.plot(sample_number, bagging_alg3_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B1={B1_list[ind1]}, k={k_list[ind2]}')
            ax.plot(sample_number, bagging_alg3_trick_obj_avg[ind1][ind2], marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3_trick, B1={B1_list[ind1]}, k={k_list[ind2]}')
    
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective', size = 20)
    ax.legend(fontsize = 'small')
    if name == None:
        fig_name = "obj_avg_" + str(B1_list) + '_' +str(k_list) + ".png"
    else:
        fig_name = name + "_obj_avg" + ".png"
    plt.savefig(fig_name)
    return

def plot_CI_test(SAA_obj_list, bagging_alg3_obj_list, bagging_alg3_trick_obj_list, sample_number, k_list, B1_list, name = None):
    number_of_iterations = len(SAA_obj_list[0])
    _, ax = plt.subplots()
    for ind1, B1 in enumerate(B1_list):
        for ind2, k in enumerate(k_list):
            if B1 == "X" or k == "X":
                continue
            diff_Bk = []
            for i in range(len(sample_number)):
                diff_Bk.append([bagging_alg3_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
            
            diff_Bk_mean = np.mean(diff_Bk, axis = 1)
            diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
            diff_conf_int = 1.96 * diff_Bk_std_err

            ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B1={B1}, k={k}')
            ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)
            
            diff_Bk = []
            for i in range(len(sample_number)):
                diff_Bk.append([bagging_alg3_trick_obj_list[ind1][ind2][i][j] - SAA_obj_list[i][j] for j in range(number_of_iterations)])
            
            diff_Bk_mean = np.mean(diff_Bk, axis = 1)
            diff_Bk_std_err = stats.sem(diff_Bk, axis = 1)
            diff_conf_int = 1.96 * diff_Bk_std_err

            ax.plot(sample_number, diff_Bk_mean, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3_trick, B1={B1}, k={k}')
            ax.fill_between(sample_number, diff_Bk_mean - diff_conf_int, diff_Bk_mean + diff_conf_int, alpha = 0.2)
    
    ax.axhline(0, color='grey', linewidth=2, linestyle='--') 
    ax.set_xlabel('Number of samples', size = 20)
    ax.set_ylabel('Objective Difference', size = 20)
    ax.legend(fontsize = 'small')
    if name == None:
        fig_name = "obj_CI_" + str(B1_list) + '_' + str(k_list) + ".png"
    else:
        fig_name = name + "_obj_CI" + ".png"
    plt.savefig(fig_name)
    return

def plot_twoPhase_cdf(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, sample_number, B_list, k_list, B12_list, name = None, skip_alg1 = False):
    fig, ax = plt.subplots(nrows=len(sample_number), figsize=(4, len(sample_number) * 3))

    def getCDF(sequence):
        xList = []
        yList = []
        for num in sorted(sequence):
            if len(xList) == 0:
                xList.append(num)
                yList.append(1 / len(sequence))
            elif num > xList[-1]:
                xList.append(num)
                yList.append(yList[-1] + 1 / len(sequence))
            else:
                yList[-1] += 1 / len(sequence)
        
        tailList = []
        for i in range(len(yList)):
            if i == 0:
                tailList.append(1)
            else:
                tailList.append(1 - yList[i - 1])

        return xList, tailList

    for i in range(len(sample_number)):
        xList, yList = getCDF(SAA_obj_list[i])
        ax[i].plot(xList, yList, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
        if not skip_alg1:
            for ind1, B in enumerate(B_list):
                for ind2, k in enumerate(k_list):
                    if B == "X" or k == "X":
                        continue
                    xList, yList = getCDF(bagging_alg1_obj_list[ind1][ind2][i])
                    ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg1, B={B_list[ind1]}, k={k_list[ind2]}')
        
        for ind1, B12 in enumerate(B12_list):
            for ind2, k in enumerate(k_list):
                if B12 == "X" or k == "X":
                    continue
                xList, yList = getCDF(bagging_alg3_obj_list[ind1][ind2][i])
                ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg3, B12={B12_list[ind1]}, k={k_list[ind2]}')
                xList, yList = getCDF(bagging_alg4_obj_list[ind1][ind2][i])
                ax[i].plot(xList, yList, marker = 's', markeredgecolor = 'none', linestyle = 'solid', linewidth = 2, label = f'Alg4, B12={B12_list[ind1]}, k={k_list[ind2]}')
        
        if i == len(sample_number) - 1:
            ax[i].set_xlabel('Objective')
        ax[i].set_ylabel('tail prob')
        ax[i].set_yscale('log')
        ax[i].set_title(f'sample size = {sample_number[i]}')
        ax[i].legend(fontsize = 'small')
    if name == None:
        fig_name = "tail_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + ".png"
    else:
        fig_name = name + "_tail" + ".png"
    fig.savefig(fig_name, dpi=600)
    return