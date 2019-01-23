# Omid55
from wordcloud import WordCloud
from langdetect import detect
import pandas as pd
import re
from langdetect import detect_langs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import string
from scipy import stats
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import enchant
import collections
import nxpd
import sys
import time as tm
import os
import itertools
import warnings
import pickle as pk
import datetime
import config
import calendar
import random
import math
import glob
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import sklearn.preprocessing as prp
warnings.filterwarnings('ignore')



class Preprocessing():

    def __init__(self):
        self.data = []
        self.external_data = []
        self.params = config.MyConfig()
        self.load_all_terminologies()
        self.assign_index_to_traders()  # creating index for traders


    def save_it(self, pkl_file_path='codes/data_saved/my_object.pkl', protocol=pk.HIGHEST_PROTOCOL):
        with open(pkl_file_path, 'wb') as handle:
            pk.dump(self, handle, protocol=protocol)


    """load the preprocessed data if it is precomputed from file, or run preprocessing on raw files"""
    def load_preprocessed_data(self, with_external = True, data_file_path = 'codes/data_saved/Categorized_messages.csv', data_file_path_for_external = 'codes/data_saved/Categorized_messages_external.csv'):
        if os.path.exists(data_file_path):
            # If the data is not saved, then load the data
            self.data = pd.DataFrame.from_csv(data_file_path)
            if with_external:
                self.external_data = pd.DataFrame.from_csv(data_file_path_for_external)
            found_preprocessed_data = True
        else:
            # If the data is not saved before, then computing the preprocessed data (files' content)
            self.preprocess_the_data(with_external)
            found_preprocessed_data = False
        print('Data: ', self.data.shape)
        if with_external:
            print('Ex Data: ', self.external_data.shape)
        return found_preprocessed_data


    """It assigns an index to every trader"""
    def assign_index_to_traders(self):
        # traders maps: trader_name to an integer index
        self.traders = {}
        index = 0
        for trader_name in self.trader_names:
            self.traders[trader_name] = index
            index += 1


    """It loads all of ANEW emotion words"""
    def init_emotions(self):
        anew_dictionary_filepath = '../BagofWords/ANEW_stemmed.csv'
        anew = pd.read_csv(anew_dictionary_filepath)
        self.anew_dict = {}
        for i in range(len(anew)):
            self.anew_dict[anew.ix[i]['Description']] = anew.ix[i]['Valence Mean']


    # Loading the terminologies
    def load_all_terminologies(self):
        # trading terms
        self.trading_terms = self.get_terms('../BagofWords/tradingterms.csv')

        # IM terms
        self.im_terms = self.get_terms('../BagofWords/IM_terms.csv')

        # traders' first name
        self.first_names = self.get_terms('../data/Information/firstnames.csv')

        # relationship terms
        tmp = self.get_terms('../BagofWords/relationship_terms.csv')
        self.relationship_terms = [rt+' ' for rt in tmp]

        # HTML codes
        self.html_codes = self.get_terms('../BagofWords/html_codes.txt')

        # traders' names
        self.trader_names = self.get_terms('../data/Information/tradernames.csv')

        # trading strategy terms
        self.trading_strategy = self.get_terms('../BagofWords/Finance markers/tradingstrategy.csv')

        # trading symbols
        self.trading_symbols = ['$', '%', '~', '<', '>']

        # stock tickers
        self.stock_tickers = []
        # ALL stock tickers which have at least 2 letters length
        # then for those stock tickers which are English words, we save them as ALL CAPS
        # for those which are not we save them all small letters and will check all situations for them

        stock_tickers_pkl_filepath = 'data_saved/Stock_tickers.pkl'
        if os.path.exists(stock_tickers_pkl_filepath):
            # loading the stock tickers preprocessing result
            f = open(stock_tickers_pkl_filepath, 'rb')
            tmp = pk.load(f)
            self.stock_tickers = tmp['stock_tickers']
            self.allstock2idx = tmp['allstock2idx']
            self.idx2allstocks = tmp['idx2allstocks']
        else:
            with open('../BagofWords/StockTickers.csv') as f:
                slines = f.readlines()
                for line in slines[2:]:
                    # line = line.lower()
                    names = line.split(',')
                    stk = names[0].strip()
                    if len(stk) >= 2:
                        if self.my_lang_check(stk):
                            self.stock_tickers.append(stk.upper())
                        else:
                            self.stock_tickers.append(stk.lower())
                self.allstock2idx = {}
                self.idx2allstocks = []
                for index, s in enumerate(self.stock_tickers):
                    self.allstock2idx[s] = index
                    self.idx2allstocks.append(s)
                self.idx2allstocks = np.array(self.idx2allstocks)
            # saving the stock tickers preprocessing result
            tmp = {'stock_tickers': self.stock_tickers, 'allstock2idx': self.allstock2idx, 'idx2allstocks': self.idx2allstocks}
            with open(stock_tickers_pkl_filepath, 'wb') as handle:
                pk.dump(tmp, handle, protocol=pk.HIGHEST_PROTOCOL)


    """It processes the content and cleans it, and returns the trading category"""
    def process_content(self, content, TRADING_TERM_THRESHOLD = 3):
        trading_related_terms = self.trading_terms
        trading_strategy_terms = self.stock_tickers + self.trading_strategy
        translator = str.maketrans('', '', string.punctuation)
        original_content = content
        content = content.lower()

        category = 'Social'
        #         # removing messages which are longer than a threshold named BROADCAST_LIMIT since we think they are broadcast messages
        #         if len(content) >= BROADCAST_TEXT_LENGTH_THRESHOLD:
        #             continue
        # removing auto messages (IM messages)
        if any(s in content for s in self.im_terms):
            return None, None
        # removing all html codes
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&amp;', '&')
        # removing tag <a> and its content
        a_index = content.find('<a')
        if a_index >= 0:
            close_a_index = content.find('</a>')
            if close_a_index == -1:
                # print('ERROR in finding closed tag </a>.')
                return None, None
            content = content[:a_index] + content[close_a_index+5:]
        ## for the rest of html codes, just remove them
        for h in self.html_codes:
            content = content.replace(h, '')
        # if it is just a number or Nan which is also a float, just remove it
        content = content.strip()
        if not content or isinstance(content, float):
            return None, None
        # check for:
        #  all non-english stock tickers (capital or small)
        #  only for all capital stock tickers which are english words
        # finding stock tickers and trading related terms
        if self.can_any_be_found(original_content, self.stock_tickers) or \
            self.can_any_be_found(content, self.stock_tickers) or \
                any(sym in content for sym in self.trading_symbols):
            category = 'Trading'
        else:
            if self.hasNumbers(content) or self.can_any_be_found(content, trading_related_terms):
        #             content_no_punc = content.translate(translator).strip()
        #             for term in self.trading_related_terms:
        #                 if term in content_no_punc:
        #                     idx = content_no_punc.index(term)
        #                     idx_end = idx+len(term)
        #                     if (idx==0 or not content_no_punc[idx-1:idx].isalpha()) and (idx_end==len(content_no_punc)-1 or not content_no_punc[idx_end:idx_end+1].isalpha()):
        ##            # re (REG EXP) works pretty good but it is really slow
        ##             pattern = r'(?<!\S){0}(?!\S)'.format(re.escape(term))
        ##             if bool(re.search(pattern, content_no_punc)):
                category = 'Trading'
        # it is trading, now classifying in more refined manner
        if category == 'Trading':
            # checking if is advice seeking
            if '?' in content:
                category += ', Advice seeking'
            # checking if it is individual trading strategy
            content_no_punc = original_content.translate(translator).strip()  # we use original content because it is not lowered,
                                                                            # in fact we have some stock tickers UPPER because they are english words
            trading_term_count = 0
            for ts in trading_strategy_terms:
                if self.can_be_found(content_no_punc, ts):
                    trading_term_count += 1
                    if trading_term_count >= TRADING_TERM_THRESHOLD:
                        category += ', Trading strategy'
                        break
            # other checks
            # ...
        return content, category


    """Main function for loading all files and preprocess the data"""
    def preprocess_the_data(self, with_external=True, day_file_path = '../data/IM/im_1_day.txt',
            time_file_path = '../data/IM/im_2_time.txt', sender_file_path = '../data/IM/im_3_sender.txt', 
            receiver_file_path = '../data/IM/im_4_receiver.txt', content_file_path = '../data/IM/im_content.txt', TRADING_TERM_THRESHOLD = 3):
        
        day_lines = open(day_file_path).readlines()
        day_lines = [it.strip() for it in day_lines if len(it.strip())>0]
        time_lines = open(time_file_path).readlines()
        time_lines = [it.strip() for it in time_lines if len(it.strip())>0]
        sender_lines = open(sender_file_path).readlines()
        sender_lines = [it.strip() for it in sender_lines if len(it.strip())>0]
        receiver_lines = open(receiver_file_path).readlines()
        receiver_lines = [it.strip() for it in receiver_lines if len(it.strip())>0]
        content_lines = open(content_file_path).readlines()
        content_lines = [it.strip() for it in content_lines]
        print('day_lines:', len(day_lines))

        # ANEW
        self.init_emotions()

        # ------------------------------------------------------------
        # # Preprocessing the data line by line
        # Removing all duplicate messages
        entire_dt = []
        for row_index, (day, time, sender, receiver, content) in enumerate(zip(day_lines, time_lines, sender_lines, receiver_lines, content_lines)):
            entire_dt.append([day, time, sender, receiver, content])
        entire_data_duplicates_dropped = pd.DataFrame(entire_dt, columns=['Date', 'Time', 'Sender', 'Receiver', 'Content'])
        entire_data_duplicates_dropped.drop_duplicates(keep='first', inplace=True)

        print(len(day_lines) - len(entire_data_duplicates_dropped), 'identical messaages were removed.')

        # deleting unuseful variables
        del(day_lines)
        del(time_lines)
        del(sender_lines)
        del(receiver_lines)
        del(content_lines)
        
        ## Process row by row
        starting_time = tm.time()

        names = self.trader_names + self.first_names
        dt = []
        external_dt = []
        sid = SentimentIntensityAnalyzer()
        
        entire_matrix_duplicates_dropped = entire_data_duplicates_dropped.as_matrix()
        del(entire_data_duplicates_dropped)
        leng01 = int(len(entire_matrix_duplicates_dropped) / 10)

        print('progress line (in 10%):')
        for counter, (day, time, sender, receiver, content) in enumerate(entire_matrix_duplicates_dropped):
            if not content or not sender or not receiver or not day:
                continue
            if not with_external and ((sender not in self.trader_names) or (receiver not in self.trader_names)):
                continue
            if sender == receiver:
                continue
            if (sender not in self.trader_names) and (receiver not in self.trader_names):
                continue
            if not counter % leng01:
                print('.', end='')

            # computing content and other info about it for all messages =>
            original_content = content
            content, category = self.process_content(content, TRADING_TERM_THRESHOLD)
            if not category:
                continue
            
            content_sentiment = sid.polarity_scores(content)['compound']
            emotion_no, mean_emotion = self.get_sentence_mean_emotion(content)
            # if both sender and receiver were inside the firm  (just inside messages)
            data_row = [day, time, sender, receiver, content, category, content_sentiment, str(self.is_english(content)), emotion_no, mean_emotion]
            if (sender in self.trader_names) and (receiver in self.trader_names):  # there are few messages from one person to him/herself which we withdraw
                dt.append(data_row)
            if ((sender in self.trader_names) and (receiver not in self.trader_names)) or ((sender not in self.trader_names) and (receiver in self.trader_names)):
                external_dt.append(data_row)
        self.data = pd.DataFrame(dt, columns=['Date', 'Time', 'Sender', 'Receiver', 'Content', 'Category', 'Sentiment', 'IsEnglish', 'EmotionalWordCount', 'MeanEmotion'])
        self.external_data = pd.DataFrame(external_dt, columns=['Date', 'Time', 'Sender', 'Receiver', 'Content', 'Category', 'Sentiment', 'IsEnglish', 'EmotionalWordCount', 'MeanEmotion'])
        del(entire_matrix_duplicates_dropped)
        print('\nDone in %.2f seconds.' % float(tm.time() - starting_time))


    """Saving the data within the object"""
    def save_the_data(self, file_path, file_path_for_external):
        pd.DataFrame.to_csv(self.data, file_path)
        print('Data with shape of', self.data.shape, 'has been saved in', file_path, 'successfully.')
        pd.DataFrame.to_csv(self.external_data, file_path_for_external)
        print('External data with shape of', self.external_data.shape, 'has been saved in', file_path_for_external, 'successfully.')


    def save_it(self, pkl_file_path='data_saved/myobj.pkl'):
        with open(pkl_file_path, 'wb') as handle:
            pk.dump(self, handle, protocol=pk.HIGHEST_PROTOCOL)


    def load_it(self, pkl_file_path='data_saved/myobj.pkl'):
        f = open(pkl_file_path, 'rb')
        return pk.load(f)


    """It computes all metrics for an array of matrices"""
    def compute_metrics_for_all_adjacency_matrices(self):
        metrics = {}
        binarize_it = True
        metrics['avgerage betweenness'] = np.zeros(len(self.weighted_adjacency_matrices))
        metrics['avgerage in degree'] = np.zeros(len(self.weighted_adjacency_matrices))
        metrics['avgerage out degree'] = np.zeros(len(self.weighted_adjacency_matrices))
        metrics['avgerage closeness'] = np.zeros(len(self.weighted_adjacency_matrices))
        metrics['algebraic connectivity of largest connected component'] = np.zeros(len(self.weighted_adjacency_matrices))
        metrics['avgerage clustering coefficient'] = np.zeros(len(self.weighted_adjacency_matrices))
        metrics['transitivity'] = np.zeros(len(self.weighted_adjacency_matrices))
        metrics['avgerage harmonic centrality'] = np.zeros(len(self.weighted_adjacency_matrices))
    #     metrics['avgerage eigenvector centrality'] = np.zeros(len(weighted_adjacency_matrices))
        for i, A in enumerate(self.weighted_adjacency_matrices):
            DG = nx.DiGraph(A)
            G = nx.Graph(A)
            GCC = max(nx.connected_component_subgraphs(G), key = len)
            metrics['avgerage betweenness'][i] = np.mean(list(nx.betweenness_centrality(G).values()))
            metrics['avgerage in degree'][i] = np.mean(list(nx.in_degree_centrality(DG).values()))
            metrics['avgerage out degree'][i] = np.mean(list(nx.out_degree_centrality(DG).values()))
            metrics['avgerage closeness'][i] = np.mean(list(nx.degree_centrality(G).values()))
            metrics['algebraic connectivity of largest connected component'][i] = nx.algebraic_connectivity(GCC)
            metrics['avgerage clustering coefficient'][i] = nx.average_clustering(G)
            metrics['transitivity'][i] = nx.transitivity(DG)
            metrics['avgerage harmonic centrality'][i] = np.mean(list(nx.harmonic_centrality(G).values()))
    #         metrics['avgerage eigenvector centrality'][i] = np.mean(list(nx.eigenvector_centrality(G).values()))
        return metrics


    """Remove all the broadcasts messages in the data with a threshold. for BROADCAST_THRESHOLD if set to 1 then it removes all broadcast messages"""
    def remove_broadcasts(self, BROADCAST_THRESHOLD=None):
        self.apply_removing_broadcast_on(self.data, BROADCAST_THRESHOLD=BROADCAST_THRESHOLD)
        print('Data after broadcast removal: ', self.data.shape)
        self.apply_removing_broadcast_on(self.external_data, BROADCAST_THRESHOLD=BROADCAST_THRESHOLD)
        print('External data after broadcast removal: ', self.external_data.shape)


    def apply_removing_broadcast_on(self, input_data, BROADCAST_THRESHOLD=None):
        # sorting the data based on date, sender and content to make sure that we can iterate over senders 
        data_sorted = input_data.sort_values(['Date', 'Sender', 'Content'])
        if not BROADCAST_THRESHOLD:
            BROADCAST_THRESHOLD = int(self.params.get('BROADCAST_THRESHOLD'))
        date = time = 0
        first = True
        same_time_messages = []
        to_be_removed_indices = []
        sender = data_sorted.ix[0]['Sender']
        content = data_sorted.ix[0]['Content']
        for index, row in data_sorted.iterrows():
            if sender == row['Sender'] and content == row['Content']:
                same_time_messages.append(index)
            else:
                if len(same_time_messages) > BROADCAST_THRESHOLD:
                    to_be_removed_indices.extend(same_time_messages)
                sender = row['Sender']
                content = row['Content']
                same_time_messages = [index]
        if len(same_time_messages) > BROADCAST_THRESHOLD:
            to_be_removed_indices.extend(same_time_messages)
        # dropping the indices that are classified as broadcasts 
        print(len(to_be_removed_indices), 'broadcast messages are getting removed.')
        input_data.drop(input_data.index[to_be_removed_indices], inplace=True)
        input_data.reset_index(inplace=True, drop=True)
        return input_data


    # REMOVE IT FROM HERE BEGINS
    def add_months(self, sourcedate, months):
        month = sourcedate.month - 1 + months
        year = int(sourcedate.year + month / 12 )
        month = month % 12 + 1
        day = min(sourcedate.day, calendar.monthrange(year,month)[1])
        return datetime.date(year,month,day)


    def intersect(self, a, b):
        return list(set(a) & set(b))

    def hasNumbers(self, input_str):
        return any(c.isdigit() for c in input_str)


    def can_any_be_found(self, content, terms):
        translator = str.maketrans('', '', string.punctuation)
        content_no_punc = content.translate(translator).strip()
        for term in terms:
            if self.can_be_found(content_no_punc, term):
                return True
        return False
    #         if term in content_no_punc:
    #             idx = content_no_punc.index(term)
    #             idx_end = idx+len(term)
    #             if (idx==0 or not content_no_punc[idx-1:idx].isalpha()) and (idx_end==len(content_no_punc)-1 or not content_no_punc[idx_end:idx_end+1].isalpha()):
    #                 return True
    #     return False


    def can_be_found(self, content, term):
        if term in content:
            idx = content.index(term)
            idx_end = idx+len(term)
            if (idx==0 or not content[idx-1:idx].isalpha()) and (idx_end>len(content)-1 or not content[idx_end].isalpha()):
                return True
        return False


    """find out if a sentence is not english"""
    def my_lang_check(self, word):
        lang = enchant.Dict("en_US")
        if word in ['lol', 'ok', 'okey', 'idk', 'sory', 'thx', 'tnx', 'tanx', 'thanx', 'hah', 'heh', 'hurtin', 'ic', 
        'chillin', 'sux', 'nah', 'faggot', 'haha', 'nothin', 'np', 'me2', 'tv', 'lookin', 'yo', 'hmm', 'loveit', 'nutz']:
            return True
        return lang.check(word) or lang.check(word.capitalize())


    def is_english(self, sentence):
        non_english = 0
        lang = enchant.Dict("en_US")
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        sentence = sentence.lower().translate(translator).strip()
        sentence = ' '.join(sentence.split())
        words = sentence.split()
        digits = 0
        for word in words:
            if word:
                if word.isdigit():
                    digits += 1
                    continue
                if not self.my_lang_check(word):
                    non_english += 1
        if not len(words) - digits:
            return True
        non_english_ratio = float(non_english) / (len(words) - digits)
    #     print(non_english_ratio)
        return non_english_ratio < 0.8
    # REMOVE IT FROM HERE ENDS


    def show_graph(self, nxgraph, just_gcc=False, pos=None):
        plt.figure()
        if not just_gcc:
            G = nxgraph
        else:
            G = max(nx.connected_component_subgraphs(nxgraph), key = len)
        nx.draw(G, pos=pos, with_labels=True)
        print('Graph GCC size: ', len(G))
        plt.show()
        print('\n')


    def show_nxpd_graph(self, nxgraph, just_gcc=False):
        if not just_gcc:
            G = nxgraph
        else:
            G = max(nx.connected_component_subgraphs(nxgraph), key = len)
        print('Graph size: ', len(G))
        return nxpd.draw(G, show='ipynb')


    def show_graph_with_nxpd(self, i, just_gcc=False):
        print('In the period of', self.periods[i][0], 'to', self.periods[i][1], ': ')
        return self.show_nxpd_graph(self.directed_graphs[i], just_gcc=just_gcc)


    def get_connected_components(self, nxgraph):
    #     GG = nx.Graph(adjacency_matrix)
    #     GG.remove_nodes_from(nx.isolates(GG))
        return sorted(nx.connected_components(nxgraph), key = len, reverse=True)


    def get_terms(self, terms_file_path):
        terms = []
        with open(terms_file_path) as f:
            terms_f = f.readlines()
            for line in terms_f:
                terms.append(line.rstrip().lower())
        return terms


    def plot_word_cloud(self, text):
        wordcloud = WordCloud().generate(text)
        # Display the generated image:
        # the matplotlib way:
        plt.imshow(wordcloud)
        plt.axis("off")
        # take relative word frequencies into account, lower max_font_size
        wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()


    def plot_metric(self, values, ylabel, periods):
        plt.plot(values)
        plt.ylabel(ylabel)
        # seting xticks
        ax = plt.axes()
        ax.set_xticks(range(len(periods)))
        labels = [p[0]+' to '+p[1] for p in periods]
        ax.set_xticklabels(labels, rotation=45)
        plt.show()


    def spliting_in_periods(self, input_data, start_date=datetime.date(2007, 1, 1), end_date=datetime.date(2010, 5, 1), step_in_months=3):
        periods = []
        current_date = start_date
        while current_date < end_date:
            current_period_end = self.add_months(current_date, step_in_months)
            periods.append([str(current_date).replace('-', ''), str(current_period_end).replace('-', '')])
            current_date = current_period_end
        indices_in_periods = collections.defaultdict(list)
        for index, data in enumerate(input_data['Date']):
            for period_index, period in enumerate(periods):
                if int(period[0]) <= int(data) < int(period[1]):
                    indices_in_periods[period_index].append(index)
                    break
        return indices_in_periods, periods


    def extract_networks(self, start_date=datetime.date(2007, 1, 1), end_date=datetime.date(2010, 5, 1), step_in_months=3, just_social_networks=True):
        if just_social_networks:
            self.original_weighted_adjacency_matrices = self.get_extracted_networks(start_date=start_date, end_date=end_date, step_in_months=step_in_months, category='Social')
        else:
            self.original_weighted_adjacency_matrices = self.get_extracted_networks(start_date=start_date, end_date=end_date, step_in_months=step_in_months)


    def get_extracted_networks(self, start_date=datetime.date(2007, 1, 1), end_date=datetime.date(2010, 5, 1), step_in_months=3, category=None):
        messages_in_periods, periods = self.spliting_in_periods(input_data=self.data, start_date=start_date, end_date=end_date, step_in_months=step_in_months)
        # network adjacency matrices
        adjacency_matrices = []
        n = len(self.traders)
        for key in messages_in_periods.keys():
            # for every period we plot a network
            A = np.zeros((n,n))
            for message_index in messages_in_periods[key]:
                # just for social messages
                #   If 2 people sends social texts to each other enough amount of messages, we consider they are friend
                if (not category) or (self.data.ix[message_index]['Category'][:6] == category):
                    sender_index = self.traders[self.data.ix[message_index]['Sender']]
                    receiver_index = self.traders[self.data.ix[message_index]['Receiver']]
                    A[sender_index, receiver_index] += 1
            adjacency_matrices.append(A)
        return adjacency_matrices


    def stats_of_networks(self):
        # edges and nodes of evolving networks
        es = []
        no = []
        for i in range(len(self.periods)):
            es.append(len(self.directed_graphs[i].edges()))
            no.append(len(self.directed_graphs[i].nodes()))
        plt.plot(es)
        # seting xticks
        ax = plt.axes()
        ax.set_xticks(np.array(range(len(self.periods)))-0.5)
        labels = [p[0]+' to '+p[1] for p in self.periods]
        ax.set_xticklabels(labels, rotation=45);
        plt.title('Edges')
        plt.figure()
        plt.plot(no)
        # seting xticks
        ax = plt.axes()
        ax.set_xticks(range(len(self.periods)))
        labels = [p[0]+' to '+p[1] for p in self.periods]
        ax.set_xticklabels(labels, rotation=45);
        plt.title('Nodes');
        # check how spare these networks are
        for i in range(len(self.periods)):
            e = len(self.directed_graphs[i].edges())
            n = len(self.directed_graphs[i].nodes())
            print('#edges/#nodes:', round(float(e)/n,2), '\t#nodes:', n, '\t#edges:', e)


    """Based on table on: http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf
    Ztest is basically computing zscore for a hypothesis:
    z = (x - mu)  /  (sigma / sqrt(n))
    Then we look up z values in ZTable. First decimal digit and first digit after decimal point should be
    found on left and second digit after decimal point on top. Then for negative values of z, we have to use:
    z >= 1.96 or z <= -1.96
    """
    def get_siginificant_matrix(self, w, all_w_rands, all_w2_rands, runs):
        n, _ = w.shape
        mu = all_w_rands / runs
        E2 = all_w2_rands / runs
        se = np.sqrt((E2 - mu * mu) / runs)  # sigma^2 = E(X^2) - E^2(X),  standard error (se) = sigma/sqrt(n)
        se[se==0] = 1   # WE SET ALL THOSE PLACES WHICH ARE 0 TO 1 TO MAKE THEM NOT PROBLEMATIC FOR DIVISION
        z = (w - mu) / se
        # idx = np.where((z < -1.96)  |  (z > 1.96))   # 95% confidence interval
        idx = np.where((z <= -2.58)  |  (z >= 2.58))   # 99% confidence interval
        significant_w = np.zeros((n,n))
        x = idx[0]
        y = idx[1]
        for i in range(len(x)):
            significant_w[x[i]][y[i]] = w[x[i]][y[i]]
        return significant_w


    def print_significant_stats(self, w, significant_w):
        n,_ = w.shape
        print('Different edge weights:', np.sum(significant_w - w))
        print('\nEdge weights which were removed:')
        ws = []
        ww = []
        c = 0
        for i in range(n):
            for j in range(n):
                if significant_w[i][j] != w[i][j]:
                    # print(w[i][j])
                    c += 1
                    ws.append(w[i][j])
                else:
                    if w[i][j] != 0:
                        ww.append(w[i][j])
        print('Number of different edges:', c)
        print('Significant:', ww)
        print('Not significant:', ws)
        # plt.hist(ws)
        # plt.show()


    def apply_significance_test_WS_rewiring(self, w, WITH_FIGURE=False):
        runs = 10000
        n,_ = w.shape
        all_w_rands = np.zeros((n,n))
        all_w2_rands = np.zeros((n,n))
        for r in range(runs):
            w_rand = np.zeros((n,n))
            for i in range(n):
                nodes = list(range(0,n))
                del(nodes[i])
                for j in range(n):
                    if w[i][j]:
                        w_rand[i][random.choice(nodes)] = w[i][j]
            all_w_rands += w_rand
            all_w2_rands += np.power(w_rand, 2)
        if WITH_FIGURE:
            sns.heatmap(all_w_rands/runs, cmap=sns.cubehelix_palette(8))
            plt.show()
        significant_w = self.get_siginificant_matrix(w, all_w_rands, all_w2_rands, runs)
        self.print_significant_stats(w, significant_w)
        return significant_w


    # two different ways to remove unsignificant edges (messages between 2 people)
    """if CHATS_THRESHOLD<0, it uses the significance test
    if CHATS_THRESHOLD==0, it basically uses nothing and considers all messages as edge (like no threshold is happened)
    if CHATS_THRESHOLD>0, then it uses the provided threshold.
    """
    def get_processed_networks(self, adjacency_matrices, CHATS_THRESHOLD=-1, confidence=99, remove_isolated_nodes_in_graphs=False):
        if not hasattr(self, 'trading_people_list'):
            raise ValueError('First run fuction: compute_profits')
        directed_graphs = []
        weighted_adjacency_matrices = []
        if CHATS_THRESHOLD >= 0:
            # applying constant threshold
            n,_ = adjacency_matrices[0].shape
            for original in adjacency_matrices:
                thresholded = np.zeros((n,n))
                for i in range(n):
                    for j in range(n):
                        if i!=j and original[i][j] >= CHATS_THRESHOLD:
                            thresholded[i][j] = original[i][j]
                # weighted_adjacency_matrices.append(thresholded)
                DG = nx.DiGraph(thresholded)
                if remove_isolated_nodes_in_graphs:
                    DG.remove_nodes_from(list(nx.isolates(DG)))
                directed_graphs.append(DG)
        else:
            # applying significance test
            for adjacency_matrix in adjacency_matrices:
                weighted_adjacency_matrices.append(self.apply_significance_test_WS_rewiring(adjacency_matrix, confidence))
                print('\n')
            for w in weighted_adjacency_matrices:
                DG = nx.DiGraph(w)
                if remove_isolated_nodes_in_graphs:
                    DG.remove_nodes_from(list(nx.isolates(DG)))
                directed_graphs.append(DG)
        for period_index, directed_graph in enumerate(directed_graphs):
            directed_graph.remove_nodes_from(list( set(range(len(self.traders))) - set(self.trading_people_list[period_index]) ))
        return directed_graphs


    def apply_significance_test_on_networks(self, remove_isolated_nodes_in_graphs=False, confidence=99):
        if not hasattr(self, 'original_weighted_adjacency_matrices'):
            raise ValueError('You should run extract_networks function first.')
        self.directed_graphs = \
            self.get_processed_networks(adjacency_matrices=self.original_weighted_adjacency_matrices, \
                confidence=confidence, remove_isolated_nodes_in_graphs=remove_isolated_nodes_in_graphs)


    def apply_threshold_on_networks(self, remove_isolated_nodes_in_graphs=False, CHATS_THRESHOLD=None):
        if not hasattr(self, 'original_weighted_adjacency_matrices'):
            raise ValueError('You should run extract_networks function first.')
        n = len(self.traders)
        if not CHATS_THRESHOLD:
            self.applied_chats_threshold = int(self.params.get('CHATS_THRESHOLD'))
        else:
            self.applied_chats_threshold = CHATS_THRESHOLD
        print('Chats threshold is going to be:', self.applied_chats_threshold)
        self.directed_graphs = \
            self.get_processed_networks(adjacency_matrices=self.original_weighted_adjacency_matrices, \
                CHATS_THRESHOLD=self.applied_chats_threshold, remove_isolated_nodes_in_graphs=remove_isolated_nodes_in_graphs)


    def keep_trading_people(self, period_index, list_of_nodes):
        if period_index<0 or period_index>len(self.periods):
            raise ValueError('Error in parameters.')
        self.directed_graphs[period_index].remove_nodes_from(list( set(range(len(self.traders))) - set(list_of_nodes) ))


    # core_just_from_GCC: if core is defined just from people in the GCC of every period
    def compute_core(self, core_just_from_GCC=False, ALL_CONNECTED_CORE=False, start_period=None, end_period=None):
        n, _ = self.weighted_adjacency_matrices[0].shape
        self.core = range(n)
        if core_just_from_GCC:
            for cc in list_of_connected_components:
                self.core = self.intersect(self.core, cc[0])
        else:
            # for all people who have sent social messages above threshold and [most importantly] were not isolated node
            if start_period:
                grs = self.directed_graphs[start_period:end_period]    # core could be computed only on a few middle months
            else:
                grs = self.directed_graphs   #[3:-3]    # core could be computed only on a few middle months
            print('#quarters:', len(grs))
            for graph in grs:
                self.core = self.intersect(self.core, graph.nodes())
                if ALL_CONNECTED_CORE:
                    self.core = list(list(nx.connected_components(graph.subgraph(self.core).to_undirected()))[0])
        print('#people in core: ', len(self.core))
        print('core: ', self.core)


    """make the weighted matrix binary using any value above treshold (exclusive) 1,
     if threshold is 0 then it converts anything larger than 0 to 1"""
    def make_matrix_binary(self, matrix, threshold=0):
        n, m = matrix.shape
        for i in range(n):
            for j in range(m):
                if matrix[i,j] > threshold:
                    matrix[i,j] = 1
        return matrix


    """It checks every triad C(n,3) of nodes in the graph
    and counts every possible one and divide by the number of 
    all possible triads. They (triads_type_counts) sum up to 1.
    It also computes person balance metric and person transitivity::
    This function computes for every individual CLASSICAL balanceness metric in every period (network).
    balanceness for node i is defined off of C(n-1,2) triads that he/she is involved in, how many of those
    are balanced (from triad300 or triad102).
    The result of this function is saved in self.person_balanceness which is a matrix of balanaceness
    for every person in every period (network).
    """
    def get_counted_total_triads_ratio(self, directed_graph):
        list_of_nodes = directed_graph.nodes()
        triads_type_counts = collections.defaultdict(int)
        person_balanceness = collections.defaultdict(int)
        person_transitivity = collections.defaultdict(int)
        person_count = collections.defaultdict(int)
        triads = list(itertools.combinations(list_of_nodes, 3))
        for triad in triads:
            tmp = directed_graph.subgraph(triad)
            triad_subgraph_matrix = nx.adjacency_matrix(tmp).todense()
            triad_subgraph_key = str(self.make_matrix_binary(triad_subgraph_matrix))
            if triad_subgraph_key not in self.triad_types:
                print(triad, 'is not found.')
                print('binary subgraph was:', triad_subgraph_key)
                print('subgraph was: ', triad_subgraph_matrix)
            triad_type_index = self.triad_types[triad_subgraph_key]
            triads_type_counts[triad_type_index] += 1
            for person in triad:
                person_count[person] += 1
                if triad_type_index <= 1:   # classical balance theory
                    person_balanceness[person] += 1
                if triad_type_index <= 8:   # relaxed balance theory with only transitive triads
                    person_transitivity[person] += 1
        for key in triads_type_counts.keys():
            triads_type_counts[key] /= len(triads) #all_possible_triads
        for person in person_count.keys():
            person_balanceness[person] /= person_count[person]
            person_transitivity[person] /= person_count[person]
        return triads_type_counts, person_balanceness, person_transitivity   # now this is a distribution, they sum up to 1


    def count_total_triads_ratio_in_all_matrices(self):
        self.count_of_triads = []
        self.person_balanceness = []
        self.person_transitivity = []
        for directed_graph in self.directed_graphs:
            triads_type_counts, balanceness, transitivity = self.get_counted_total_triads_ratio(directed_graph)
            self.count_of_triads.append(triads_type_counts)
            self.person_balanceness.append(balanceness)
            self.person_transitivity.append(transitivity)
            print('.', end=' ')


    def show_dynamic_of_triads_ratio(self, triads_indices=range(16)):
        if len(triads_indices) > 16:
            raise ValueError('We do not have more than 16 triads, len(triads_indices) =', len(triads_indices) )
        plt.figure()
        # seting xticks
        ax = plt.axes()
        ax.set_xticks(range(len(self.periods)))
        labels = [p[0]+' to '+p[1] for p in self.periods]
        ax.set_xticklabels(labels, rotation=45)
        plt.ylabel('Ratio')
        filled_markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', '|', '_']
        legend_names = []
        for i, triad_num in enumerate(triads_indices):
            triad_num_counts = []
            for period in self.count_of_triads:
                triad_num_counts.append(period[triad_num])
            legend_names.append('Triad ' + self.triad_types_labels[triad_num])
            plt.plot(triad_num_counts[:-1], marker=filled_markers[i])
        plt.legend(legend_names);


    """Initialize 16 triads based on Noah's paper"""
    def initialize_triads(self):
        self.triad_types_list = [
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],  # Triad label 300
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],  # Triad label 102
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Triad label 003
            [[0, 0, 1], [1, 0, 1], [1, 0, 0]],  # Triad label 120D
            [[0, 1, 1], [0, 0, 0], [1, 1, 0]],  # Triad label 120U
            [[0, 1, 1], [0, 0, 1], [0, 0, 0]],  # Triad label 030T
            [[0, 0, 0], [1, 0, 1], [0, 0, 0]],  # Triad label 021D
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],  # Triad label 021U
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # Triad label 012
            [[0, 1, 0], [0, 0, 1], [0, 0, 0]],  # Triad label 021C
            [[0, 1, 0], [1, 0, 1], [0, 0, 0]],  # Triad label 111U
            [[0, 1, 0], [1, 0, 0], [0, 1, 0]],  # Triad label 111D
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # Triad label 030C
            [[0, 1, 1], [1, 0, 0], [1, 0, 0]],  # Triad label 201
            [[0, 1, 0], [1, 0, 1], [1, 0, 0]],  # Triad label 120C
            [[0, 1, 0], [1, 0, 1], [1, 1, 0]]   # Triad label 210
        ]
        self.triad_types_labels = ['300', '102', '003', '120D', '120U', '030T', '021D', '021U',
                                   '012', '021C', '111U', '111D', '030C', '201', '120C', '210']
        self.triad_types = {}
        for index, triad_type in enumerate(self.triad_types_list):
            for permutation in self.get_all_triads_permutations(triad_type).keys():
                self.triad_types[permutation] = index
        self.triad_permutations = []
        for triad in self.triad_types_list:
            self.triad_permutations.append(self.get_all_triads_permutations(triad))


    """swap two nodes in a matrix and return the resulting matrix"""
    def swap_nodes_in_matrix(self, matrix, n1, n2):
        tmp = np.copy(matrix)
        tmp[:, [n1, n2]] = tmp[:, [n2, n1]]
        tmp[[n1, n2], :] = tmp[[n2, n1], :]
        return tmp


    """get all of permutations of nodes in a matrix. It computes different matrices
    with swapping same columns and rows."""
    def get_all_triads_permutations(self, triad_matrix):
        permutations = [triad_matrix]
        mat01 = self.swap_nodes_in_matrix(triad_matrix, 0, 1)
        mat02 = self.swap_nodes_in_matrix(triad_matrix, 0, 2)
        mat12 = self.swap_nodes_in_matrix(triad_matrix, 1, 2)
        permutations.extend([mat01, mat02, mat12])
        permutations.extend(
            [self.swap_nodes_in_matrix(mat01, 0, 2),
             self.swap_nodes_in_matrix(mat01, 1, 2),
             self.swap_nodes_in_matrix(mat02, 0, 1),
             self.swap_nodes_in_matrix(mat02, 1, 2),
             self.swap_nodes_in_matrix(mat12, 0, 1),
             self.swap_nodes_in_matrix(mat12, 0, 2)])
        # result = []
        # for permutation in permutations:
        #     key = str(np.array(permutation, dtype=float))
        #     if key not in result:
        #         result.append(key)
        # return result
        result = {}
        for permutation in permutations:
            key = str(np.array(permutation, dtype=float))
            if key not in result:
                result[key] = permutation
        return result


    """average of an array, if array is empty average is 0"""
    def my_mean(self, a):
        if not a:
            return 0
        return np.mean(a)


    """Find subgraph oftriad_nodes in directed graph of w, make it binary 
    and map it to one of triads and return its index"""
    def get_triad_key_binarized_from_graph(self, DG, triad_nodes):
        tmp = DG.subgraph(triad_nodes)
        triad_subgraph_matrix = nx.adjacency_matrix(tmp).todense()
        triad_subgraph_key = str(self.make_matrix_binary(triad_subgraph_matrix))
        if triad_subgraph_key not in self.triad_types:
            print(triad_nodes, 'is not found.')
            print('binary subgraph was:', triad_subgraph_key)
            print('subgraph was: ', triad_subgraph_matrix)
        triad_type_index = self.triad_types[triad_subgraph_key]
        return triad_type_index


    def plot_triad_ratios(self):
        # nn = math.ceil(np.sqrt(len(self.periods)))
        sns.set(rc={'figure.figsize': (30,30)})
        labels = [p[0]+' to '+p[1] for p in self.periods]
        _, axarr = plt.subplots(4, 4)
        for triad_num in range(len(self.triad_types_labels)):
            i1 = int(triad_num / 4)
            i2 = triad_num % 4
            triad_num_counts = []
            for period in self.count_of_triads:
                triad_num_counts.append(period[triad_num])
            sns.regplot(x=np.array(range(len(triad_num_counts))), y=np.array(triad_num_counts), ax=axarr[i1,i2])
            axarr[i1,i2].set_xticks(range(len(self.periods)))
            if i1 == 3:
                axarr[i1,i2].set_xticklabels(labels, rotation=45)
            if i2 == 0:
                axarr[i1,i2].set_ylabel('Ratio')
            axarr[i1,i2].title.set_text('Triad ' + self.triad_types_labels[triad_num])
        plt.show()


    # MARKOVIAN TRANSITION MATRIX
    # ---------------------------------------------------------------------------------------
    """ Based on page 7 in paper titled "A stochastic model for change in group structure".
    I compute P here as transition_matrix:"""
    # MARKOVIAN TRANSITION MATRIX
    # ---------------------------------------------------------------------------------------
    """ Based on page 7 in paper titled "A stochastic model for change in group structure".
    I compute P here as transition_matrix:"""
    # MARKOVIAN TRANSITION MATRIX
    # ---------------------------------------------------------------------------------------
    """ Based on page 7 in paper titled "A stochastic model for change in group structure".
    I compute P here as transition_matrix:"""
    def compute_markov_transition_matrix(self, WITH_NORMALIZATION=True, REMOVE_003=False, WITH_FIGURE=True, figure_dimensions=None):
        # if not hasattr(self, 'transition_matrix'):
        tn = len(self.triad_types_labels)
        pn = len(self.periods)
        self.transition_matrix = np.zeros((tn, tn))
        self.transition_matrices_in_periods = [np.zeros((tn,tn)) for i in range(pn-1)]
        for i in range(pn-1):   # for between every 2 period
            g1 = self.directed_graphs[i]
            g2 = self.directed_graphs[i+1]
            print('.', end='')
            nodes1 = set(g1.nodes())
            nodes2 = set(g2.nodes())
            list_of_intersected_nodes = list(nodes1.intersection(nodes2))
            triads = list(itertools.combinations(list_of_intersected_nodes, 3))
            for triad in triads:
                triad_index_i = self.get_triad_key_binarized_from_graph(g1, triad)
                triad_index_iplus1 = self.get_triad_key_binarized_from_graph(g2, triad)
                self.transition_matrix[triad_index_i, triad_index_iplus1] += 1
                self.transition_matrices_in_periods[i][triad_index_i, triad_index_iplus1] += 1
        if REMOVE_003:
            self.transition_matrix[:,2] = 0
            for transition_matrix_in_periods in self.transition_matrices_in_periods:
                transition_matrix_in_periods[:,2] = 0
        title = 'Count Transition Matrix'
        if WITH_NORMALIZATION:
            title = 'Probability Transition Matrix'
            # normalize all items in each line separately by sum
            self.transition_matrix = (self.transition_matrix.T / np.sum(self.transition_matrix, axis=1)).T
            self.transition_matrix = np.nan_to_num(self.transition_matrix)
            for i in range(len(self.transition_matrices_in_periods)):
                sums = np.sum(self.transition_matrices_in_periods[i], axis=1)
                sums[sums==0] = 1  # for those rows which are all 0s, just to not produce nans
                self.transition_matrices_in_periods[i] = (self.transition_matrices_in_periods[i].T / sums).T
            if WITH_FIGURE:
                # figure of all periods' transition matrix
                nn = math.ceil(np.sqrt(len(self.periods)))
                if figure_dimensions:
                    n1 = figure_dimensions[0]
                    n2 = figure_dimensions[1]
                else:
                    n1 = n2 = nn
                sns.set(rc={"figure.figsize": (n2*10,n1*10)})
                f, axarr = plt.subplots(n1, n2)
                for i in range(pn-1):
                    i1 = int(i / n2)
                    i2 = i % n2
                    sns.heatmap(self.transition_matrices_in_periods[i], ax=axarr[i1,i2], cmap=sns.cubehelix_palette(8))
                    axarr[i1,i2].title.set_text('network of ' + self.periods[i][0] + '-' + self.periods[i][1] + '  to  network of ' + self.periods[i+1][0] + '-' + self.periods[i+1][1])
                    # axarr[i1,i2].title.set_text('('+chr(i+ord('A'))+')')
                    axarr[i1,i2].set_xticks(np.array(range(16))+0.5, minor=True)
                    axarr[i1,i2].set_yticks(np.array(range(16))+0.5, minor=True)
                    axarr[i1,i2].set_xticklabels(self.triad_types_labels, rotation=90)
                    axarr[i1,i2].set_yticklabels(self.triad_types_labels, rotation=0)
            plt.show()
        if WITH_FIGURE:
            # figure of the whole transition matrix
            sns.set(rc={"figure.figsize": (8,8)})
            plt.title(title)
            sns.heatmap(self.transition_matrix, cmap=sns.cubehelix_palette(8))        
            plt.xlabel('Triads')
            plt.ylabel('Triads')
            ax = plt.axes()
            ax.set_xticks(np.array(range(16))+0.5, minor=True)
            ax.set_yticks(np.array(range(16))+0.5, minor=True)
            ax.set_xticklabels(self.triad_types_labels, rotation=90)
            ax.set_yticklabels(self.triad_types_labels, rotation=0)
            plt.show()


    # DISTRIBUTION OF TRIADS
    # -------------------------------------------------------------------------
    """Test it with the follwing examples:
        # w = [[0,0,0],[0,0,0],[0,0,0]]
        w = [[0,1,0],[0,0,0],[0,0,0]]

        # w = [[0,100,100],[0,0,98],[99,0,0]]
        # w = [[0,10,190],[0,0,98],[99,0,0]]

        # w = [[0,1,100],[0,0,98],[99,0,0]]
        # w = [[0,50,51],[50,0,48],[49,50,0]]

        # w = [[0,100,100],[100,0,100],[100,1,0]]
        prob = self.compute_probability_of_membership_to_every_triad(w)
        print(max(prob), min(prob), np.std(prob))
        print(prob)
    """
    def compute_probability_of_membership_to_every_triad(self, subgraph):
        if len(subgraph) != 3:
            raise ValueError('Subgraph should be triad (graph of 3 nodes). However, it has', len(subgraph), 'nodes.')
        probabilty = np.zeros(len(self.triad_permutations))
        if not np.sum(subgraph):    # if subgraph is all zeros
            return np.zeros(16)     # it does not count anything for Triad 003
        for index, triads in enumerate(self.triad_permutations):
            metrics = []
            for triad in triads.values():
                overlapping_edges = []
                nonoverlapping_edges = []
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            continue
                        if triad[i][j]:
                            overlapping_edges.append(subgraph[i][j])
                        else:
                            nonoverlapping_edges.append(subgraph[i][j])
                metric = self.my_mean(overlapping_edges) - self.my_mean(nonoverlapping_edges)
                metrics.append(metric)
            probabilty[index] = max(metrics)
        probabilty -= np.min(probabilty)
        if np.sum(probabilty):
            probabilty /= np.sum(probabilty)
        # if np.sum(subgraph) > 0:
        #     print('Subgraph:', subgraph)
        #     print('Probability:', probabilty)
        #     print('\n\n')
        return probabilty


    """It checks every triad C(n,3) of nodes in the graph
    and computes the probability that evey subgraph of 3 belongs
    to each triad and records the probability."""
    def compute_total_probability_of_membership_to_all_triads(self, original_weighted_adjacency_matrix, list_of_nodes):
        zscore_normalization = True
        if zscore_normalization:  # zscore normalization
            # applying Zscore to the weights first
            m = original_weighted_adjacency_matrix
            mf = m.flatten()
            # mf = np.delete(mf, np.where(mf==0))
            W = (m - np.mean(mf)) / np.std(mf)
            # removing those which are further than std_num_threshold standard deviation from mean
            std_num_threshold = 3   # 3std includes 99.7% of data
            W[(W>std_num_threshold)|(W<-std_num_threshold)] = 0
        else:  # no normalization
            W = original_weighted_adjacency_matrix
        triads_type_probs = np.zeros(len(self.triad_types_labels))
        triads = list(itertools.combinations(list_of_nodes, 3))
        for triad in triads:
            triad_subgraph = W[triad,:][:,triad]
            all_possible_triads = len(triads)
            triads_type_probs += self.compute_probability_of_membership_to_every_triad(triad_subgraph) / all_possible_triads
        return triads_type_probs


    def compute_total_probability_of_membership_to_all_triads_in_all_matrices(self, REMOVE_ISOLATED_NODES=False):
        print('compute_total_probability_of_membership_to_all_triads_in_all_matrices:')
        self.probability_of_triads = []
        for originalw in self.original_weighted_adjacency_matrices:
            if REMOVE_ISOLATED_NODES:
                DG = nx.DiGraph(originalw)
                DG.remove_nodes_from(list(nx.isolates(DG)))
                list_of_nodes = DG.nodes()
            else:
                list_of_nodes = range(0, originalw.shape[0])
            self.probability_of_triads.append(self.compute_total_probability_of_membership_to_all_triads(originalw, list_of_nodes))
            print('.', end=' ')
        # FIGURE: a line for every period
        plt.figure()
        # seting xticks
        ax = plt.axes()
        ax.set_xticks(range(len(self.triad_types_labels)))
        labels = ['Triad ' + triad_num for triad_num in self.triad_types_labels]  #[p[0]+' to '+p[1] for p in self.periods]
        ax.set_xticklabels(labels, rotation=45)
        plt.ylabel('Probability')
        filled_markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', '|', '_']
        for i in range(len(self.periods)):
            plt.plot(self.probability_of_triads[i], marker=filled_markers[i%len(filled_markers)])
        plt.legend([p[0]+' to '+p[1] for p in self.periods])
        plt.show()
        # FIGURE: a line for every triad
        plt.figure()
        # seting xticks
        ax = plt.axes()
        ax.set_xticks(range(len(self.periods)))
        labels = [p[0]+' to '+p[1] for p in self.periods]
        ax.set_xticklabels(labels, rotation=45)
        plt.ylabel('Probability')
        # filled_markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', '|', '_']
        for i, triad_num in enumerate(self.triad_types_labels):
            vec = []
            for j in range(len(self.periods)):
                vec.append(self.probability_of_triads[j][i])
            plt.plot(vec, marker=filled_markers[i%len(filled_markers)])
        plt.legend(['Triad ' + triad_num for triad_num in self.triad_types_labels]);
        plt.show()


    def get_stationary_distribution(self, matrix):
        eps = 0.0001
        while eps < 1:
            A = matrix.copy()
            A = np.nan_to_num(A)
            A += eps
            A = (A.T / np.sum(A, axis=1)).T
            values, vectors = np.linalg.eig(A.T)
            F = [x.real for x in vectors[:,0]] #np.array([float(v[0]) for v in vectors])
            F /= sum(F)      # it should be probability, since this is just a vector we can multiply it with any scalar
            if any([f < 0 for f in F]):
                eps *= 10
            else:
                break
        return F


    def print_stationary_state(self, matrix):
        F = self.get_stationary_distribution(matrix)
        for i in range(16):
            print("Triad{0}:\t{1:.2f}".format(self.triad_types_labels[i], F[i]))
            

    def print_my_stationary_distribution(self, A):
        eps = 0.0001
        A = np.nan_to_num(A)
        A += eps
        A = (A.T / np.sum(A, axis=1)).T
        n = len(A)
        s = np.ones(n) * (1/n)
    #     s = np.zeros(n)
    #     s[0] = 1
        eps = 0.00000001
        cnt = 1000
        converged = False
        for i in range(cnt):
            s_new = np.dot(s, A)
            if np.linalg.norm(s_new-s) < eps:
                converged = True
                print('It is converged in', i, 'rounds.')
                break
            s = s_new
        if converged:
            for i in range(16):
                print("Triad{0}:\t{1:.4f}".format(self.triad_types_labels[i], s[i]))
        else:
            print('Not converged.')


    def plot_stationary_state_for_periods(self, profit_labels=None):
        stationary_distributions = np.zeros((len(self.periods)-1, len(self.triad_types_labels)))
        ylabels = []
        for i in range(len(self.periods)-1):
            if profit_labels:
                ylabels.append(self.periods[i][0]+'-'+self.periods[i][1] + ' -> ' + self.periods[i+1][0]+'-'+self.periods[i+1][1]+' : ' + profit_labels[i])
            else:
                ylabels.append(self.periods[i][0]+'-'+self.periods[i][1] + ' -> ' + self.periods[i+1][0]+'-'+self.periods[i+1][1])
            P = self.get_stationary_distribution(self.transition_matrices_in_periods[i])
            stationary_distributions[i,:] = P
        sns.heatmap(stationary_distributions, cmap=sns.cubehelix_palette(8))
        # seting xticks
        ax = plt.axes()
        ax.set_yticks(np.array(range(len(ylabels)))+0.5, minor=True)
        ax.set_yticklabels(reversed(ylabels), rotation=0)
        ax.set_xticks(np.array(range(len(self.triad_types_labels)))+0.5, minor=True)
        ax.set_xticklabels(self.triad_types_labels, rotation=0);
        # np.savetxt('StationaryDistributions.csv', stationary_distributions, fmt='%.4f', delimiter=',')
        plt.show()


    # -------------------------------------------------------------------------------------
    # Performance of day traders codes
    # -------------------------------------------------------------------------------------

    def load_trading_data(self, trading_data_filepath = 'data_saved/all_trading_data.pkl', FORCE=False):
        if not FORCE and os.path.exists(trading_data_filepath):
            handle = open(trading_data_filepath, 'rb')
            trading_data = pk.load(handle)
        else:
            # Day trading data loading from csv_all folder
            trading_data = pd.DataFrame()
            for foldername in glob.glob('../data/Trades/csv_all/*'):
                username = foldername[foldername.rfind('/')+1:]
                user_dt = pd.DataFrame()
                for file in glob.glob(foldername+'/*.csv'):
                    f = pd.read_csv(file)
            #         date_in_chats_format = pd.Series([x[6:] + x[:2] + x[3:5] if not isinstance(x, float) else '-1' for x in f['Trade Date']], name='Date')
                    date_in_chats_format = pd.Series([x[6:] + x[:2] + x[3:5] for x in f['Trade Date'] if not isinstance(x, float)], name='Date')
                    tmp = []
                    for x in f['Execution Time']:
                        if isinstance(x, float):
                            tmp.append('-1')
                        else:
                            t = x.split(':')
                            tmp.append(str(t[0]).zfill(2) + str(t[1]).zfill(2) + str(t[2].split(' ')[0]).zfill(2))
                            # print(tmp)
                    time_in_chats_format = pd.Series(tmp, name='Time')
                    user_dt = user_dt.append(
                        pd.concat([
                                date_in_chats_format, 
                                time_in_chats_format,
                                f['Symbol/Description'],
                                f['Settle CCY Principal/Gross Notional Value'],
                                f['Quantity']
                                # add more features if you need
                                # ...
                            ], axis=1)
                        )
                user_dt['Username'] = username
                trading_data = trading_data.append(user_dt)
            trading_data.sort_values(['Date', 'Time'], inplace=True)
            trading_data.reset_index(drop=True, inplace=True)
            trading_data.rename(columns={'Symbol/Description': 'Stock', 'Settle CCY Principal/Gross Notional Value': 'Money'}, inplace=True)
    #         del(trading_data['index'])
            trading_data = trading_data[['Date', 'Time', 'Username', 'Stock', 'Money', 'Quantity']]
            # droping nans
            trading_data.dropna(inplace=True)
            trading_data.reset_index(drop=True, inplace=True)
            # saving the trading data
            with open(trading_data_filepath, 'wb') as handle:
                pk.dump(trading_data, handle)
        return trading_data


    # dailyprofit.Portfolio are not unique, they have all stocks that have been traded
    def load_daily_portfolio_profits(self, trading_daily_profit_filepath='data_saved/dailyprofit.pkl', FORCE=False):
        if not FORCE and os.path.exists(trading_daily_profit_filepath):
            handle = open(trading_daily_profit_filepath, 'rb')
            self.dailyprofit = pk.load(handle)
        else:
            dt = []
            self.trading_data = self.load_trading_data(FORCE=FORCE)
            for name, group in self.trading_data.groupby(['Username', 'Date']):
                stock_list = []
                for x in np.unique(list(group['Stock'])):
                    if isinstance(x, float):
                        continue
                    x = x.lower().strip()
                    x = x.split(' ')[0]
                    if x[0].isdigit():
                        continue
                    if x not in self.allstock2idx:   # if stock is not defined in our StockTickers.csv 
                        continue
                    stock_list.append(x)
                if stock_list:
                    dt.append( [name[1], name[0], stock_list, group['Money'].sum(), abs(group['Money']).sum(), abs(group['Quantity']).sum()] )
            self.dailyprofit = pd.DataFrame(dt, columns=['Date', 'Username', 'Portfolio', 'Money', 'AbsTrades', 'AbsStockQuantity'])
            self.dailyprofit.sort_values(['Date'], inplace=True)
            self.dailyprofit.reset_index(inplace=True)
            del(self.dailyprofit['index'])
            # saving the daily profit
            with open(trading_daily_profit_filepath, 'wb') as handle:
                pk.dump(self.dailyprofit, handle)
            self.dailyprofit.to_csv('data_saved/DailyProfits.csv')



    # def get_Wrand(W):
#     Wrand = np.zeros(W.shape)
#     n, _ = W.shape
#     C = np.sum(W, axis=0)
#     R = np.sum(W, axis=1)
#     counter = 0
#     prev = 0
#     while np.sum(C) + np.sum(R) > 0:
#         i = np.random.choice(range(n) , p=(R/np.sum(R)))
#         j = np.random.choice(range(n) , p=(C/np.sum(C)))
#         if i!=j and not Wrand[i,j]:
#             Wrand[i,j] = 1
#             R[i] -= 1
#             C[j] -= 1
#         ## << CHECK HERE >> FIX IT IT HERE
#         counter += 1
#         if not counter % 100:
#             if np.sum(C) + np.sum(R) == prev:
#                 break
#             else:
#                 prev = np.sum(C) + np.sum(R)
#         ## << CHECK HERE >> FIX IT IT HERE
#     return Wrand


    # def get_sentence_mean_emotion(self, anew, sentence):
    #     translator = str.maketrans('', '', string.punctuation)
    #     sentence_no_punc = sentence.translate(translator).strip()
    #     words = sentence_no_punc.split(' ')
    #     count = 0
    #     emotion_score = 0
    #     for word in words:
    #         index = np.where(anew.Description == word)[0]
    #         if len(index):
    #             if len(index) > 1:
    #                 print('This word is repeated in the dictionary: ', word)
    #                 index = index[0]
    #             emotion_score += anew.ix[index[0]]['Valence Mean']
    #             count += 1
    #     if count:
    #         emotion_score /= count
    #     return count, emotion_score


    def get_my_stem(self, word):
        p_stemmer = PorterStemmer()
        try:
            return p_stemmer.stem(word)
        except:
            return word


    def get_sentence_mean_emotion(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        en_stop = get_stop_words('en')
        stopped_tokens = [w for w in tokens if not w in en_stop]
        stemmed_words = [self.get_my_stem(w) for w in stopped_tokens]
        count = 0
        emotion_score = 0
        for word in stemmed_words:
            if word in self.anew_dict:
                emotion_score += self.anew_dict[word]
                count += 1
        if count:
            emotion_score /= count
        return count, emotion_score


    # ------------------------------------------------------------
    # MOTIF CODES begin
    # ------------------------------------------------------------
    # generating random networks for a given SWITCH_COUNT_COEFFICIENT using a given adjancecy matrix A.
    #  randomly switching single and double edges until at least LEN time nothing changes
    def generate_random_network_for_motifs(self, A, SWITCH_COUNT_COEFFICIENT=300):
        RA = A.copy()
        n, _ = A.shape
        edge_count = len(np.where(A>0)[0])
        LEN = 1000   # if after LEN times, it couldn't switch, we call it consensus (convergence) and terminate the algorithm
        switching_count = 0
        prev_switching_count = 0
        desired_switching_count = SWITCH_COUNT_COEFFICIENT * edge_count
        counter = 0

        # for num in range(SWITCH_COUNT_COEFFICIENT * edge_count):
        while switching_count < desired_switching_count:
            # randomly choose 2 edges
            counter += 1

            # double edges
            double_edgesRA = np.floor((RA + RA.T)/2)
            dx, dy = np.where(double_edgesRA>0)
            i, j = np.random.choice(range(len(dx)), size=2, replace=False)
            s1 = dx[i]
            t1 = dy[i]
            s2 = dx[j]
            t2 = dy[j]
            if not (RA[s1, t2] or RA[s2, t1] or RA[t2, s1] or RA[t1, s2] or s1==t2 or s1==s2 or s2==t1 or t1==t2 ):
                RA[s1, t1] = 0
                RA[t1, s1] = 0
                RA[s2, t2] = 0
                RA[t2, s2] = 0
                RA[s1, t2] = 1
                RA[t2, s1] = 1
                RA[s2, t1] = 1
                RA[t1, s2] = 1
                switching_count += 1

            # single edges
            double_edgesRA = np.floor((RA + RA.T)/2)   # we have to compute it again because RA might have been changed
            single_edgesRA = RA - double_edgesRA
            sx, sy = np.where(single_edgesRA>0)
            i, j = np.random.choice(range(len(sx)), size=2, replace=False)
            s1 = sx[i]
            t1 = sy[i]
            s2 = sx[j]
            t2 = sy[j]
            if not( RA[s1, t2] or RA[s2, t1] or s1==t2 or s1==s2 or s2==t1 or t1==t2 ):
                RA[s1, t1] = 0
                RA[s2, t2] = 0
                RA[s1, t2] = 1
                RA[s2, t1] = 1
                switching_count += 1
            
            if not counter % LEN:
                if prev_switching_count == switching_count:
                    print('Not converged ...')
                    RA = []
                    break
                else:
                    prev_switching_count = switching_count
        return RA


    # counting the number of each 16 triads in the binary matrix A
    #   it returns distribution of each triad count, every person's balanceness and every person's transitivity
    #      (A must be binary matrix)
    def count_triad_motifs(self, A, COMPUTE_TRIADS_RATIO=True):
        n, _ = A.shape
        triads_type_counts = np.zeros(16)
        person_balanceness = np.zeros(n)
        person_transitivity = np.zeros(n)
        person_count = np.zeros(n)
        triads = list(itertools.combinations(range(n), 3))
        for triad in triads:
            triad_subgraph = A[triad,:][:,triad]
            triad_subgraph_key = str(triad_subgraph)
            if triad_subgraph_key not in self.triad_types:
                print(triad, 'is not found.')
                print('binary subgraph was:', triad_subgraph)
                print('subgraph was: ', weighted_adjacency_matrix[triad,:][:,triad])
            triad_type_index = self.triad_types[triad_subgraph_key]
            triads_type_counts[triad_type_index] += 1
            for person in triad:
                person_count[person] += 1
                if triad_type_index <= 1:   # classical balance theory
                    person_balanceness[person] += 1
                if triad_type_index <= 8:   # relaxed balance theory with only transitive triads
                    person_transitivity[person] += 1
        if COMPUTE_TRIADS_RATIO:
            triads_type_counts /= len(triads)   # all_possible_triads
        for person in range(n):
            if person_count[person]:
                person_balanceness[person] /= person_count[person]
                person_transitivity[person] /= person_count[person]
        return triads_type_counts, person_balanceness, person_transitivity   # now this is a distribution, they sum up to 1


    # How significant each triad is in each period (how many std are they far from mean) REAL - RANDOM
    def triads_significance_test(self, STD_COUNT_THRESHOLD=5, WITH_PRINT=False):
        # plotting a figure about how significant each triad is in each period
        distance_from_mean = np.zeros((len(self.periods), 16))
        for j in range(len(self.triads_type_counts_list)):
            r = (self.real_triads_type_counts_list[j] - self.triads_type_counts_list[j].mean(axis=0)) / (0.00001 + self.triads_type_counts_list[j].std(axis=0))
            r[np.where(abs(r)<STD_COUNT_THRESHOLD)] = 0
            distance_from_mean[j,:] = r
        sns.heatmap(distance_from_mean, cmap=sns.cubehelix_palette(8))
        # seting xticks
        ax = plt.axes()
        ax.set_yticks(np.array(range(len(self.periods)))+0.5, minor=True)
        # ax.set_yticklabels(reversed([p[0]+' to '+p[1] for p in self.periods]), rotation=0)
        ax.set_yticklabels(['Quarter ' + str(i+1) for i in range(len(self.periods))], rotation=0)
        ax.set_xticks(np.array(range(len(self.triad_types_labels)))+0.5, minor=True)
        ax.set_xticklabels(self.triad_types_labels, rotation=0);
        # np.savetxt('StationaryDistributions.csv', stationary_distributions, fmt='%.4f', delimiter=',')
        plt.show()
        if WITH_PRINT:
            # printing how significant each triad is in each period
            for j in range(len(self.triads_type_counts_list)):
                r = (self.real_triads_type_counts_list[j] - self.triads_type_counts_list[j].mean(axis=0)) / (0.00001 + self.triads_type_counts_list[j].std(axis=0))
                print('Period: ', self.periods[j][0], '->', self.periods[j][1], ' :')
                for i in range(16):
                    if abs(r[i]) >= STD_COUNT_THRESHOLD:
                        print(self.triad_types_labels[i], '\t:', int(round(r[i])))
                print('\n')


    # How significant balanceness of each trader is from average of random balancenesses of that trader
    def network_balance_significance_test(self, path_to_save=None):
        pn = len(self.periods)
        runs,_ = self.triads_type_counts_list[0].shape
        reals = np.zeros(pn)
        means = np.zeros(pn)
        stds = np.zeros(pn)
        avg_number_stds = 0
        for i in range(pn):
            reals[i] = (self.real_triads_type_counts_list[i][0] + self.real_triads_type_counts_list[i][1]) / self.real_triads_type_counts_list[i].sum()
            classical_balanced_metric_vector = (self.triads_type_counts_list[i][:,0] + self.triads_type_counts_list[i][:,1]) / self.triads_type_counts_list[i].sum(axis=1)
            means[i] = classical_balanced_metric_vector.mean()
            stds[i] = classical_balanced_metric_vector.std()
            nstds = (reals[i] - means[i]) / stds[i]
            avg_number_stds += nstds / pn
            print('#stds: ', nstds)
        print('Avg of #std: ', avg_number_stds)
        # figure
        plt.figure()
        plt.errorbar(range(len(means)), means, yerr=stds)
        plt.plot(reals, 'ro')
        plt.legend(['real', 'random'])
        plt.ylabel('Balance ratio')
        # plt.title('Classicial balance metric dynamics and significance test')
        ax = plt.axes()
        ax.set_xticks([-0.5] + list(range(pn)) + [pn-0.5])
        # ax.set_xticklabels([''] + [p[0]+' to '+p[1] for p in self.periods], rotation=45)
        ax.set_xticklabels([''] + ['Quarter ' + str(i) for i in range(1,1+len(self.periods))], rotation=45)
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("right")
            
        if path_to_save:
            plt.savefig(path_to_save)
        plt.show()


    # How significant balanceness of each trader is from average of random balancenesses of that trader
    #  in fact this function comptues the ratio of traders that their balancencess is significantlly different than random
    def trader_balance_significance_test(self, STD_COUNT_THRESHOLD=5):
        runs, _ = self.triads_type_counts_list[0].shape
        more_balance_ratio = np.zeros(len(self.periods))
        less_balance_ratio = np.zeros(len(self.periods))
        for j in range(len(self.real_person_balanceness_list)):   # for each period
            # r = (self.real_person_balanceness_list[j] - self.person_balanceness_list[j].mean(axis=0)) / (0.00001 + self.person_balanceness_list[j].std(axis=0))
            r = (self.real_person_balanceness_list[j] - self.person_balanceness_list[j].mean(axis=0)) / ((0.00001 + self.person_balanceness_list[j].std(axis=0))/np.sqrt(runs))  #zscore
            for i in range(len(r)):     # for each trader
                if r[i] >= STD_COUNT_THRESHOLD:
                    more_balance_ratio[j] += (1/len(r))
                if r[i] <= -STD_COUNT_THRESHOLD:
                    less_balance_ratio[j] += (1/len(r))
        return more_balance_ratio, less_balance_ratio


    # 1.96: 95%   and   2.58: 99%
    def print_ztest_triad_motifs(self):
        # printing how significant each triad is in each period
        for j in range(len(self.triads_type_counts_list)):
            real = self.real_triads_type_counts_list[j]
            runs, triads_count = self.triads_type_counts_list[j].shape
            mu = self.triads_type_counts_list[j].mean(axis=0)
            sigma = self.triads_type_counts_list[j].std(axis=0)
            zscore = (real - mu) / (0.00001 + sigma/np.sqrt(runs))
            print('Period: ', self.periods[j][0], '->', self.periods[j][1], ' :')
            for i in range(triads_count):
                if abs(zscore[i]) >= 2.58:
                    print(self.triad_types_labels[i], '\t:', int(round(zscore[i])))
            print('\n')


    def generate_and_count_motifs(self, RUNS=200, SWITCH_COUNT_COEFFICIENT=300, COMPUTE_TRIADS_RATIO=True):
        self.triads_type_counts_list = []
        self.person_balanceness_list = []
        self.person_transitivity_list = []
        self.real_triads_type_counts_list = []
        self.real_person_balanceness_list = []
        self.real_person_transitivity_list = []

        for network_index in range(len(self.periods)):
            print('.', end='')

            W_isolated_removed = nx.adjacency_matrix(self.directed_graphs[network_index]).toarray()
            W = np.zeros(W_isolated_removed.shape)
            W[np.where(W_isolated_removed>0)] = 1   # make the weighted matrix binary (binarization is needed for motif algorithm)
            n, _ = W.shape

            triads_type_counts = np.zeros((RUNS, 16))
            person_balanceness = np.zeros((RUNS, n))
            person_transitivity = np.zeros((RUNS, n))
            real_triads_type_counts, real_person_balanceness, real_person_transitivity = self.count_triad_motifs(W, COMPUTE_TRIADS_RATIO)

            for run in range(RUNS):
        #         if not run % 20:
        #             print(run)
                Wrand = []
                while not len(Wrand):
                    Wrand = self.generate_random_network_for_motifs(W, SWITCH_COUNT_COEFFICIENT)
                triads_type_counts[run,:], person_balanceness[run,:], person_transitivity[run,:] = self.count_triad_motifs(Wrand, COMPUTE_TRIADS_RATIO)

            self.triads_type_counts_list.append(triads_type_counts)
            self.person_balanceness_list.append(person_balanceness)
            self.person_transitivity_list.append(person_transitivity)
            self.real_triads_type_counts_list.append(real_triads_type_counts)
            self.real_person_balanceness_list.append(real_person_balanceness)
            self.real_person_transitivity_list.append(real_person_transitivity)


    # NOT USEFULFIGURES
    def plot_motif_differences(self):
        for i in range(len(self.triads_type_counts_list)):
            print('Period: ', self.periods[i][0], '->', self.periods[i][1], ' :')
            
            # figure
            plt.figure()
            plt.errorbar(range(self.triads_type_counts_list[i].shape[1]), np.mean(self.triads_type_counts_list[i], axis=0), yerr=np.std(self.triads_type_counts_list[i], axis=0)) #/np.sqrt(len(triads_type_counts)))
            plt.plot(self.real_triads_type_counts_list[i], 'ro')
            plt.legend(['real', 'random'])
            plt.xlabel('Triads')
            plt.ylabel('Frequency')
            plt.title('Triad frequency')
            ax = plt.axes()
            ax.set_xticks(np.array(range(16)))
            ax.set_xticklabels(self.triad_types_labels, rotation=90)
            plt.show()

            # WITHOUT 102 AND 003
            plt.figure()
            dd = np.mean(self.triads_type_counts_list[i], axis=0)
            dds = np.std(self.triads_type_counts_list[i], axis=0)
            rdd = self.real_triads_type_counts_list[i].copy()
            dd = np.delete(dd, [1,2])
            dds = np.delete(dds, [1,2])
            rdd = np.delete(rdd, [1,2])
            plt.errorbar(range(len(dd)), dd, yerr=dds) #/np.sqrt(len(triads_type_counts)))
            plt.plot(rdd, 'ro')
            plt.legend(['real', 'random'])
            plt.xlabel('Triads')
            plt.ylabel('Frequency')
            plt.title('Triad frequency')
            ax = plt.axes()
            ax.set_xticks(np.array(range(14)))
            labs = self.triad_types_labels
            labs = np.delete(labs, [1,2])
            ax.set_xticklabels(labs, rotation=90);
            plt.show()

            # figure
            plt.figure()
            plt.errorbar(range(self.person_balanceness_list[i].shape[1]), np.mean(self.person_balanceness_list[i], axis=0), yerr=np.std(self.person_balanceness_list[i], axis=0)) #/np.sqrt(len(person_balanceness)))
            plt.plot(self.real_person_balanceness_list[i])
            plt.legend(['real', 'random'])
            plt.xlabel('Traders')
            plt.ylabel('Frequency')
            plt.title('Traders balanceness')
            plt.show()

            # figure
            plt.figure()
            plt.errorbar(range(self.person_transitivity_list[i].shape[1]), np.mean(self.person_transitivity_list[i], axis=0), yerr=np.std(self.person_transitivity_list[i], axis=0)) #/np.sqrt(len(person_transitivity)))
            plt.plot(self.real_person_transitivity_list[i])
            plt.legend(['real', 'random'])
            plt.xlabel('Traders')
            plt.ylabel('Frequency')
            plt.title('Traders transitivity')
            plt.show()

            print('\n\n')


    # ------------------------------------------------------------
    # MOTIF CODES ends
    # ------------------------------------------------------------


    def apply_on_GCC(self, result_from_GCC, G):
        for node in list(set(G.nodes()) - set(result_from_GCC.keys())):
            result_from_GCC[node] = 0
        return result_from_GCC


    def get_computed_metrics_for_network(self, DG):
        metrics = {}
        # DG
        metrics['in_degree'] = nx.in_degree_centrality(DG)
        metrics['out_degree'] = nx.out_degree_centrality(DG)
        metrics['degree'] = nx.degree_centrality(DG)
        metrics['load'] = nx.load_centrality(DG)
        metrics['eigenvector'] = nx.eigenvector_centrality(DG, max_iter=10000)
        metrics['harmonic'] = nx.harmonic_centrality(DG)
        metrics['closeness'] = nx.closeness_centrality(DG)
        metrics['betweenness'] = nx.betweenness_centrality(DG)
        metrics['katz'] = nx.katz_centrality(DG)
        # G
        G = nx.to_undirected(DG)
        metrics['clustering_coefficient'] = nx.clustering(G)
        # GCC
        GCC = max(nx.connected_component_subgraphs(G), key = len)
        metrics['communicability'] = self.apply_on_GCC(nx.communicability_betweenness_centrality(GCC), G)
        metrics['current_flow_betweenness'] = self.apply_on_GCC(nx.current_flow_betweenness_centrality(GCC),G)
        metrics['current_flow_closeness'] = self.apply_on_GCC(nx.current_flow_closeness_centrality(GCC),G)
        # my balance-related metrics
        _, metrics['balance'], metrics['balance_transitivity'] = self.get_counted_total_triads_ratio(DG)
        return metrics


    def normalize_the_features(self, features, normalization_type='minmax'):
        if normalization_type == 'zscore':
            # Standard Normalization (x-mean(x) / std(x))
            standard_scaler = prp.StandardScaler()
            x_scaled = standard_scaler.fit_transform(features)
            new_features = pd.DataFrame(x_scaled, columns=features.columns)
        elif normalization_type == 'minmax':
            # MIN MAX Normalization
            min_max_scaler = prp.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(features)
            new_features = pd.DataFrame(x_scaled, columns=features.columns)
        else:
            raise ValueError('normalization_type is not valid, it is: ', normalization_type)
        return new_features


    def compute_profits(self, START_DATE, END_DATE, STEPS_IN_MONTHS=3):
        # we save summation of money that every trader made in every period in: profits
        profit_indices_in_periods, self.periods = self.spliting_in_periods(self.dailyprofit, START_DATE, END_DATE, STEPS_IN_MONTHS)
        self.profits = [{} for i in range(len(self.periods))]
        for index, period_indices in enumerate(profit_indices_in_periods.values()):
            quarter_profits = self.dailyprofit.ix[period_indices]
            for name, group in quarter_profits.groupby(['Username']):
                self.profits[index][self.traders[name]] = group['Money'].mean()
        # who are trading
        self.trading_people_list = []
        for profit in self.profits:
            self.trading_people_list.append(list(profit.keys()))


    # this function computes 3 basic info about traders in each periods and returns
    #   as a list of dictionary that each dict maps tradername to each info
    # 1. Number of instant messages sent
    # 2. Number of active trading days
    # 3. Average size of trades: absolute of trades and abs of number of stocks traded
    def get_basicinfo_about_traders(self, START_DATE=datetime.date(2007, 10, 1), END_DATE=datetime.date(2009, 4, 1), STEPS_IN_MONTHS=3):
        # self.compute_profits(START_DATE, END_DATE, STEPS_IN_MONTHS)
        # Number of instant messages sent
        messages_in_periods, _ = self.spliting_in_periods(self.data, START_DATE, END_DATE, STEPS_IN_MONTHS)
        message_count = [collections.defaultdict(lambda: 0) for i in range(len(messages_in_periods))]
        social_message_count = [collections.defaultdict(lambda: 0) for i in range(len(messages_in_periods))]
        for period_index in range(len(messages_in_periods)):
            for message_index in messages_in_periods[period_index]:
                traderID = self.traders[self.data.ix[message_index]['Sender']]
                message_count[period_index][traderID] += 1
                if self.data.ix[message_index]['Category'][:6] == 'Social':
                    social_message_count[period_index][traderID] += 1
        # Number of active trading days and Average size of trades
        profit_indices_in_periods, _ = self.spliting_in_periods(self.dailyprofit, START_DATE, END_DATE, STEPS_IN_MONTHS)
        active_trading_days = [{} for i in range(len(messages_in_periods))]
        abs_trades = [{} for i in range(len(messages_in_periods))]
        abs_quantity = [{} for i in range(len(messages_in_periods))]
        for period_index in range(len(messages_in_periods)):
            for name, group in self.dailyprofit.ix[profit_indices_in_periods[period_index]].groupby(['Username']):
                active_trading_days[period_index][self.traders[name]] = len(group)
                abs_trades[period_index][self.traders[name]] = np.mean(group['AbsTrades'])
                abs_quantity[period_index][self.traders[name]] = np.mean(group['AbsStockQuantity'])
        return {'message_count': message_count, 'social_message_count': social_message_count, 'active_trading_days': \
            active_trading_days, 'abs_trades': abs_trades, 'abs_quantity': abs_quantity}





    """This function builds the dataset from scratch as if no network has been extracted before"""
    def build_the_dataset(self, metrics_category=['social', 'financial', 'all'], controls = ['mac', 'time', 'tid'], \
        SAVE_THE_DATASET=False, STEPS_IN_MONTHS=3, CHATS_THRESHOLD=5, CONFIDENCE=99, \
        START_DATE=datetime.date(2007, 10, 1), END_DATE=datetime.date(2009, 4, 1), \
        LABELING='mean', DATASET_NORMALIZATION=False):
        
        starting_time = tm.time()

        mac = pd.read_csv('../data/Information/Macroeconomic Variables.csv')
        mac_columns = list(mac.columns[3:])

        self.compute_profits(START_DATE, END_DATE, STEPS_IN_MONTHS)
        # make labels for classification from profits
        labels = []
        for i in range(len(self.profits)):
            if LABELING == 'mean':
                # mean:
                comparison_point = np.mean(list(self.profits[i].values()))
            elif LABELING == 'median':
                # median:
                comparison_point = np.median(list(self.profits[i].values()))
            else:
                print('ERROR IN LABELING METHOD ...')

            tmp = {}
            for trader, profit in self.profits[i].items():
                if profit >= comparison_point:
                    tmp[trader] = 1
                else:
                    tmp[trader] = 0
            labels.append(tmp)
        print('extracting networks ...')
        if 'social' in metrics_category:
            social_adjacency_matrices = self.get_extracted_networks(START_DATE, END_DATE, STEPS_IN_MONTHS, 'Social')
        if 'financial' in metrics_category:
            trading_adjacency_matrices = self.get_extracted_networks(START_DATE, END_DATE, STEPS_IN_MONTHS, 'Tradin')
        if 'all' in metrics_category:
            all_comm_adjacency_matrices = self.get_extracted_networks(START_DATE, END_DATE, STEPS_IN_MONTHS)
        print('getting processed networks ...')
        # self.compute_profits(START_DATE, END_DATE, STEPS_IN_MONTHS)
        nets = {}
        if 'social' in metrics_category:
            social_nets = self.get_processed_networks(social_adjacency_matrices, CHATS_THRESHOLD, CONFIDENCE)
            nets['social'] = social_nets
        if 'financial' in metrics_category:
            trading_nets = self.get_processed_networks(trading_adjacency_matrices, CHATS_THRESHOLD, CONFIDENCE)
            nets['financial'] = trading_nets
        if 'all' in metrics_category:
            allcomm_nets = self.get_processed_networks(all_comm_adjacency_matrices, CHATS_THRESHOLD, CONFIDENCE)
            nets['all'] = allcomm_nets

        # we have 3 different networks:
        # computing the column names that we can be consistent for all samples
        self.initialize_triads()
        tmp = self.get_computed_metrics_for_network(social_nets[0])
        feature_names = sorted(tmp.keys())
        column_names = []
        for metric_category in metrics_category:
            for feature_name in feature_names:
                column_names.append(metric_category + '_' + feature_name)
        for metric_category in metrics_category:
            column_names.append(metric_category + '_' + 'balance_in_stationary')
            column_names.append(metric_category + '_' + 'transitivity_in_stationary')
        # computing transition matrices
        balance_transitivity_in_stationary = {}
        if 'social' in metrics_category:
            balance_transitivity_in_stationary['social'] = get_balance_transitivity_in_stationary(self, social_nets)
        if 'financial' in metrics_category:
            balance_transitivity_in_stationary['financial'] = get_balance_transitivity_in_stationary(self, trading_nets)
        if 'all' in metrics_category:
            balance_transitivity_in_stationary['all'] = get_balance_transitivity_in_stationary(self, allcomm_nets)
                                
        data_labels = []
        data_profits = []
        dt = []
        basic_info = self.get_basicinfo_about_traders(START_DATE, END_DATE, STEPS_IN_MONTHS)
        basic_info_columns = sorted(basic_info.keys())
        column_names.extend(basic_info_columns)
        for period_index in range(len(labels)):
            # network metrics
            mets = {}
            if 'social' in metrics_category:
                mets['social'] = self.get_computed_metrics_for_network(social_nets[period_index])
            if 'financial' in metrics_category:
                mets['financial'] = self.get_computed_metrics_for_network(trading_nets[period_index])
            if 'all' in metrics_category:
                mets['all'] = self.get_computed_metrics_for_network(allcomm_nets[period_index])
            metrics = [mets[metric_category] for metric_category in metrics_category]

            for trader, label in labels[period_index].items():
                # the code is already checked about: $trader have $label and nets[period_index][$trader] has its communications.
                sample = []
                for metric_index in range(len(metrics)):
                    for feature_name in feature_names:
                        metric = metrics[metric_index][feature_name]
                        if trader in metric:
                            sample.append(metric[trader])
                        else:
                            sample.append(0)
                data_labels.append(label)
                data_profits.append(self.profits[period_index][trader])


                # adding stationary distribution balance and transitivity
                for metric_category in metrics_category:
                    sample.append(balance_transitivity_in_stationary[metric_category][0][period_index])
                    sample.append(balance_transitivity_in_stationary[metric_category][1][period_index])

                # adding basic info
                for basic_info_column in basic_info_columns:
                    sample.append(basic_info[basic_info_column][period_index][trader])

                # ---- === CONTROL VARIABLES === ----
                if 'mac' in controls:
                    # adding Macroeconomic variables
                    for clm in mac_columns:
                        sample.append(mac.ix[period_index][clm])

                if 'time' in controls:
                    # adding time as fixed effects
                    times = np.zeros(len(labels))
                    times[period_index] = 1
                    sample.extend(times)

                if 'tid' in controls:
                    # adding trader id as fixed effects
                    tradersID = np.zeros(len(self.traders))
                    tradersID[trader] = 1
                    sample.extend(tradersID)
                # ---- === CONTROL VARIABLES === ----

                dt.append(sample)

        all_column_names = column_names
        if 'mac' in controls:
            all_column_names += mac_columns
        if 'time' in controls:
            all_column_names += ['period'+str(i) for i in range(1,len(labels)+1)]
        if 'tid' in controls:
            all_column_names += ['trader'+str(i) for i in range(1,len(self.traders)+1)]
        data_features = pd.DataFrame(dt, columns=all_column_names)
        self.dataset = {'features': data_features, 'labels': data_labels, 'profits': data_profits}

        # if you wanted to add them together it is this easy:
        # data = dataset['features'].copy()
        # data['label'] = dataset['labels']

        # normalization
        if DATASET_NORMALIZATION:
            original_features = self.dataset['features'].copy()
            self.dataset['features'] = self.normalize_the_features(original_features)
            
        # saving all data
        if SAVE_THE_DATASET:
            self.dataset['features'].to_csv('Features.csv')
            pd.DataFrame(self.dataset['labels'], columns=['label']).to_csv('Labels.csv')
            pd.DataFrame(self.dataset['profits'], columns=['profit']).to_csv('Profits.csv')

        print('\nDone in %.2f seconds.' % float(tm.time() - starting_time))
        return nets

    #  WEEK-BASED FUNCTIONS STARTS

    def compute_profits_with_weeks(self, START_DATE, END_DATE, step_in_weeks=2):
        # we save summation of money that every trader made in every period in: profits
        profit_indices_in_periods, self.periods = self.spliting_in_periods_with_weeks(self.dailyprofit, START_DATE, END_DATE, step_in_weeks)
        self.profits = [{} for i in range(len(self.periods))]
        for index, period_indices in enumerate(profit_indices_in_periods.values()):
            quarter_profits = self.dailyprofit.ix[period_indices]
            for name, group in quarter_profits.groupby(['Username']):
                self.profits[index][self.traders[name]] = group['Money'].mean()
        # who are trading
        self.trading_people_list = []
        for profit in self.profits:
            self.trading_people_list.append(list(profit.keys()))

    def spliting_in_periods_with_weeks(self, input_data, start_date=datetime.date(2007, 1, 1), end_date=datetime.date(2010, 5, 1), step_in_weeks=2):
        periods = []
        current_date = start_date
        while current_date < end_date:
            current_period_end = current_date + datetime.timedelta(step_in_weeks * 7)
            periods.append([str(current_date).replace('-', ''), str(current_period_end).replace('-', '')])
            current_date = current_period_end
        indices_in_periods = collections.defaultdict(list)
        for index, data in enumerate(input_data['Date']):
            for period_index, period in enumerate(periods):
                if int(period[0]) <= int(data) < int(period[1]):
                    indices_in_periods[period_index].append(index)
                    break
        return indices_in_periods, periods

    def extract_networks_with_weeks(self, start_date=datetime.date(2007, 1, 1), end_date=datetime.date(2010, 5, 1), step_in_weeks=2, just_social_networks=True):
        if just_social_networks:
            self.original_weighted_adjacency_matrices = self.get_extracted_networks_with_weeks(start_date=start_date, end_date=end_date, step_in_weeks=step_in_weeks, category='Social')
        else:
            self.original_weighted_adjacency_matrices = self.get_extracted_networks_with_weeks(start_date=start_date, end_date=end_date, step_in_weeks=step_in_weeks)

    def get_extracted_networks_with_weeks(self, start_date=datetime.date(2007, 1, 1), end_date=datetime.date(2010, 5, 1), step_in_weeks=2, category=None):
        messages_in_periods, periods = self.spliting_in_periods_with_weeks(input_data=self.data, start_date=start_date, end_date=end_date, step_in_weeks=step_in_weeks)
        # network adjacency matrices
        adjacency_matrices = []
        n = len(self.traders)
        for key in messages_in_periods.keys():
            # for every period we plot a network
            A = np.zeros((n,n))
            for message_index in messages_in_periods[key]:
                # just for social messages
                #   If 2 people sends social texts to each other enough amount of messages, we consider they are friend
                if (not category) or (self.data.ix[message_index]['Category'][:6] == category):
                    sender_index = self.traders[self.data.ix[message_index]['Sender']]
                    receiver_index = self.traders[self.data.ix[message_index]['Receiver']]
                    A[sender_index, receiver_index] += 1
            adjacency_matrices.append(A)
        return adjacency_matrices

    # WEEK-BASED FUNCTIONS ENDS


def get_balance_transitivity_in_stationary(pp, nets):
    pp.directed_graphs = nets
    pp.compute_markov_transition_matrix(WITH_FIGURE=False)
    balance_in_stationary = np.zeros(len(pp.periods))
    transitivity_in_stationary = np.zeros(len(pp.periods))
    for i in range(len(pp.transition_matrices_in_periods)):
        P = pp.get_stationary_distribution(pp.transition_matrices_in_periods[i])
        balance_in_stationary[i+1] = (np.sum(P[0:2]) / np.sum(P))      # we start from i+1 because we do not ->
        transitivity_in_stationary[i+1] = (np.sum(P[0:8]) / np.sum(P))   # have any transition before first period
    return [balance_in_stationary, transitivity_in_stationary]





def main():
    pp = Preprocessing()
    if not pp.load_preprocessed_data(with_external=False):  # after loading, if the preprocessed file was not found, save it
        pp.save_the_data(file_path='data_saved/Categorized_messages.csv',
                         file_path_for_external='data_saved/Categorized_external_messages.csv')


if __name__ == '__main__':
    main()
