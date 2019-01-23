# Omid55
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def show_matrix(pp, matrix, people_names=None):
    sns.set(rc={"figure.figsize": (8, 20)})
    sns.heatmap(matrix, cmap=sns.cubehelix_palette(8))
    # seting xticks
    ax = plt.axes()
    ax.set_xticks(np.array(range(len(pp.periods)))+0.5, minor=True)
    labels = [p[0]+' to '+p[1] for p in pp.periods]
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(np.array(range(len(pp.traders)))+0.5, minor=True)
    if people_names is None:
        people_names = sorted(pp.traders.keys(), reverse=True)
    ax.set_yticklabels(people_names, rotation=0)

    
def show_corr2_forlists(pp, A, B, JUST_SIGNIFICANT=True):
    sns.set(rc={"figure.figsize": (8, 8)})
    # collumn-wise correlation of balanceness and profit
    # (correlation of profit and balanceness)
    corr_balanceness_performance_overtime = []
    for period_index in range(len(pp.periods)):
        corr_balanceness_performance_overtime.append(
            sp.stats.pearsonr(A[period_index], B[period_index])
        )
    for c in range(len(corr_balanceness_performance_overtime)):
        if JUST_SIGNIFICANT and corr_balanceness_performance_overtime[c][1] > 0.05:
            corr_balanceness_performance_overtime[c] = (0,0)
    print(corr_balanceness_performance_overtime)
    plt.plot([x[0] for x in corr_balanceness_performance_overtime]);   \
        # plt.plot([x[0] for x in corr_balanceness_performance_overtime if x[1]<0.05]);
        
        
def rescale_to_range_of_0_and_1(vals):
    min_value = min(vals)
    vals -= min_value
    vals = vals / max(vals)
    return vals


def plot_regressions_of(pp, profits_avg, n1, n2, vector1, vector2, name1, name2, RESCALE_01=False, ROBUST=False, WITHOUT_ZEROS=False):
#     dt = []
    pn = len(pp.periods)
    v1 = vector1.copy()
    v2 = vector2.copy()

    sns.set(rc={"figure.figsize": (20,15)})
    _, axarr = plt.subplots(n1, n2)
    xs = []
    ys = []
    for i in range(pn):
        x = v1[i]
        y = v2[i]

        if WITHOUT_ZEROS:
            zero_indices = np.where(x == 0)
            x = np.delete(x, zero_indices[0] )
            y = np.delete(y, zero_indices[0] )
        
#         # fixed effect regression model data preparation
#         dummy = np.zeros(pn)
#         dummy[i] = 1
#         for xv, yv in zip(x, y):
#             item = [xv]
#             item.extend(dummy)
#             item.extend([yv])
#             dt.append(item)
#         # fixed effect regression model data preparation

        i1 = int(i / n2)
        i2 = i % n2
        
        xs.extend(x)
        ys.extend(y)
        
        if RESCALE_01:
            x = rescale_to_range_of_0_and_1(x.copy())
            y = rescale_to_range_of_0_and_1(y.copy())
        
        m, b = np.polyfit(x, y, 1)
        if ROBUST:
            sns.regplot(x=x, y=y, ax=axarr[i1,i2], robust=True)
        else:
            sns.regplot(x=x, y=y, ax=axarr[i1,i2])
        axarr[i1,i2].title.set_text('Slope:' + "{0:.2f}".format(m) + ',  Intercept:' + "{0:.2f}".format(b) \
                                    + ',  AverageProfit:' + "{0:.2f}K $".format(np.mean(profits_avg[i])/1000))
        if i2 == 0:
            axarr[i1,i2].set_ylabel(name2)
        if i1 == n1-1:
            axarr[i1,i2].set_xlabel(name1)
    sns.set(rc={"figure.figsize": (8,8)})
    plt.figure()
    m, b = np.polyfit(xs, ys, 1)
    plt.scatter(xs, ys)
    xs = np.array(xs)
    ys = np.array(ys)
    plt.plot(xs, m*xs + b, '-')
    plt.ylabel(name2)
    plt.xlabel(name1)
    plt.title('Aggregated figure');
    # sns.regplot(x=xs, y=ys)
    
    
