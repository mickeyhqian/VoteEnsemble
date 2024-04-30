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

def plot_optGap_twoPhase(obj_opt, minmax, SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B_list, k_list, B12_list):
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
    fig_name = "opt_gap_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + ".png"
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


def plot_twoPhase(SAA_obj_avg, bagging_alg1_obj_avg, bagging_alg3_obj_avg, bagging_alg4_obj_avg, sample_number, B_list, k_list, B12_list):
    _, ax = plt.subplots()
    ax.plot(sample_number, SAA_obj_avg, marker = 'o', markeredgecolor = 'none', color = 'blue',linestyle = 'solid', linewidth = 2, label = 'SAA')
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
    fig_name = "obj_avg_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + ".png"
    plt.savefig(fig_name)
    return

def plot_CI_twoPhase(SAA_obj_list, bagging_alg1_obj_list, bagging_alg3_obj_list, bagging_alg4_obj_list, sample_number, B_list, k_list, B12_list):
    number_of_iterations = len(SAA_obj_list[0])
    _, ax = plt.subplots()
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
    fig_name = "obj_diff_" + str(B_list) + '_' +str(k_list) + '_' + str(B12_list) + ".png"
    plt.savefig(fig_name)
    return