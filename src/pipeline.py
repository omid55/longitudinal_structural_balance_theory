# Omid55
# pipeline of entire preprocessing and processing of day trading data
import preprocessing as prep
import seaborn as sns
import numpy as np
import imp
import networkx as nx
import datetime
import scipy as sp
import matplotlib.pyplot as plt
prep = imp.reload(prep)


def execute(BROADCAST_THRESHOLD=1, step_in_months=3, just_social_networks=True, remove_isolated_nodes_in_graphs=True, 
    APPLY_THRESHOLD=True, CHATS_THRESHOLD=5, start_period=3, end_period=-3, ALL_CONNECTED_CORE=False):
    print('Execution has started :')
    print('preprocessing ...')
    pp = prep.Preprocessing()
    if not pp.load_preprocessed_data():   # after loading, if the preprocessed file was not found, save it
        pp.save_the_data(file_path = '/home/omid55/Projects/balanced_theory_study/codes/data_saved/Categorized_messages.csv')

    print('\nremoving broadcast messages ...')
    pp.remove_broadcasts(BROADCAST_THRESHOLD = BROADCAST_THRESHOLD)
    # pp.save_the_data(file_path = '/home/omid55/Projects/balanced_theory_study/codes/data_saved/Preprocessed_messages.csv')

    print('\nstatistics about messages ...')
    social_cnt = 0
    for index, row in pp.data.iterrows():
        if row['Category'] == 'Social':
            social_cnt += 1
    print('We have', len(pp.data), 'messages.')
    print('Off of those,', social_cnt, 'are social messages.')
    print(social_cnt / len(pp.data), 'is the ratio of social messages in total.\n')

    print('extracting networks ...')
    pp.extract_networks(step_in_months=step_in_months, just_social_networks=just_social_networks)

    print('\nremoving non-significant edges ...')
    if APPLY_THRESHOLD:
        pp.apply_threshold_on_networks(remove_isolated_nodes_in_graphs=remove_isolated_nodes_in_graphs, CHATS_THRESHOLD=CHATS_THRESHOLD)
    else:
        pp.apply_significance_test_on_networks(remove_isolated_nodes_in_graphs=remove_isolated_nodes_in_graphs)

    # pp = pp.load_it(pkl_file_path='data_saved/myobj.pkl')
    # mode = 1

    mode = 0
    while mode <= 1:
        if mode == 1:
            # replacing the core with all networks
            print('\n\n\n<<<<<=== RUNNING ALL EXPERIMENTS JUST FOR THE CORE ===>>>>>')
            print('computing core ...')
            pp.compute_core(ALL_CONNECTED_CORE=ALL_CONNECTED_CORE, start_period=start_period, end_period=end_period)
            pp.directed_graphs = []
            for i in range(len(pp.weighted_adjacency_matrices)):
                pp.weighted_adjacency_matrices[i] = pp.weighted_adjacency_matrices[i][pp.core,:][:,pp.core]
                DG = nx.DiGraph(pp.weighted_adjacency_matrices[i])
                DG.remove_nodes_from(nx.isolates(DG))
                pp.directed_graphs.append(DG)
            if start_period:
                pp.periods = pp.periods[start_period:end_period]
                pp.weighted_adjacency_matrices = pp.weighted_adjacency_matrices[start_period:end_period]
                pp.directed_graphs = pp.directed_graphs[start_period:end_period]

        print('\nstats about networks ...')
        pp.stats_of_networks()

        print('\ncounting triads in networks ...')
        pp.initialize_triads()
        pp.count_total_triads_ratio_in_all_matrices(REMOVE_ISOLATED_NODES=remove_isolated_nodes_in_graphs)
        pp.plot_triad_ratios()

        print('\ncomputing transition matrix ...')
        pp.compute_markov_transition_matrix(WITH_NORMALIZATION=True, REMOVE_003=False, REMOVE_ISOLATED_NODES=remove_isolated_nodes_in_graphs)
        pp.print_stationary_state(pp.transition_matrix.copy())

        print('\ncomputing transition matrices stationary distribution ...')
        sns.set(rc={"figure.figsize": (8, 8)})
        pp.plot_stationary_state_for_periods()

        print('\ncomputing distribution of triads ...')
        pp.compute_total_probability_of_membership_to_all_triads_in_all_matrices(REMOVE_ISOLATED_NODES=remove_isolated_nodes_in_graphs)

        if mode == 0:
            print('\nsaving ...')
            pp.save_it(pkl_file_path='data_saved/myobj.pkl')

            # # ------------------------------PERFORMANCE --------------------------------------------------------------------------------------
            print('Preprocessing the performance data')
            pp.load_daily_portfolio_profits()

            # start and end date for performance
            mystart_date = datetime.date(2007, 10, 1)
            myend_date = datetime.date(2009, 4, 1)
            indices_in_periods, periods = pp.spliting_in_periods(pp.dailyprofit, start_date=mystart_date, end_date=myend_date, step_in_months=3)
            pp.extract_networks(start_date=mystart_date, end_date=myend_date, step_in_months=step_in_months, just_social_networks=just_social_networks)
            pp.apply_threshold_on_networks(remove_isolated_nodes_in_graphs=remove_isolated_nodes_in_graphs, CHATS_THRESHOLD=CHATS_THRESHOLD)

            profits = np.zeros((len(pp.traders), len(periods)))
            for index, period_idx in enumerate(indices_in_periods.values()):
                quarter_profits = pp.dailyprofit.ix[period_idx]
                for name, group in quarter_profits.groupby(['Username']):
                    profits[pp.traders[name], index] = group['Money'].sum()
            profits_normalized_mean = profits / profits.mean(axis=0)

            pp.count_total_triads_ratio_in_all_matrices(REMOVE_ISOLATED_NODES=remove_isolated_nodes_in_graphs)
            # for all
            print('\nAll balanceness and profits:')
            show_corr2(pp.person_balanceness, profits_normalized_mean)
            # after removing 3 people
            people_to_be_removed = ['luchente39', 'gslgary', 'poppebill1']
            indices = [pp.traders[i] for i in people_to_be_removed]
            profits_smaller = np.delete(profits, indices, axis=0)
            balanceness_smaller = np.delete(pp.person_balanceness, indices, axis=0)
            profits_smaller_normalized_mean = profits_smaller / profits_smaller.mean(axis=0)
            print('\n3 People fewer of balanceness and profits:')
            show_corr2(balanceness_smaller, profits_smaller_normalized_mean)

        mode += 1



def show_corr2(A, B, JUST_SIGNIFICANT=True):
    sns.set(rc={"figure.figsize": (8, 8)})
    # collumn-wise correlation of balanceness and profit
    # (correlation of profit and balanceness)
    corr_balanceness_performance_overtime = []
    for period_index in range(A.shape[1]):
        corr_balanceness_performance_overtime.append(
            sp.stats.pearsonr(A[:,period_index], B[:,period_index])
        )
    for c in range(len(corr_balanceness_performance_overtime)):
        if JUST_SIGNIFICANT and corr_balanceness_performance_overtime[c][1] > 0.05:
            corr_balanceness_performance_overtime[c] = (0,0)
    print(corr_balanceness_performance_overtime)
    plt.plot([x[0] for x in corr_balanceness_performance_overtime]);   # plt.plot([x[0] for x in corr_balanceness_performance_overtime if x[1]<0.05]);


def main():
    print('Main:')
    execute()


if __name__ == '__main__':
    main()