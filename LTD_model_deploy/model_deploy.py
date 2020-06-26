import pickle
from os.path import abspath, join, dirname
import argparse

import numpy as np
import pandas as pd
import nltk
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# directory this file is located in
this_dir = dirname(abspath(__file__))

class AnalyzeEventDescription(object):

    #nb_model = pickle.load(join(this_dir,'models','tfidf_NB_model.pk', 'rb'))
    #infile = open(join(this_dir,'models','tfidf_NB_model.pk'),'rb')
    #nb_model = pickle.load(infile, encoding='bytes')


    with open(join(this_dir,'models','tfidf_NB_model2.pk'), 'rb') as fin:
        nb_model = pickle.load(fin)

    with open(join(this_dir,'models','count_vec2.pk'), 'rb') as fin:
        count_vec = pickle.load(fin)


    #lda_model = pickle.load(join(this_dir,'models','lda_model.pk'))
    #d2v_model = pickle.load(join(this_dir,'models','d2v_model.pk'))


    def __init__(self, descr_text, verbose):
        # anything dynamic you want to initialize?
        self.nb_model = AnalyzeEventDescription.nb_model
        self.count_vec = AnalyzeEventDescription.count_vec
        self.descr_text = descr_text
        self.proc_text = self._process_descr_text()

        self.verbose = verbose
        

    def _process_descr_text(self):
        # Preprocessing of text data

        # Convert all the string to lower cases
        proc_text = self.descr_text.lower()

        # \S+ means anything that is not an empty space
        proc_text = re.sub('http\S*', '', proc_text)

        # \s+ means all empty space (\n, \r, \t)
        proc_text = re.sub('\s+', ' ', proc_text)
        proc_text = re.sub('[^\w\s]', '', proc_text)

        # Adding domain-based stop words to general English stop words list and ignoring these in data
        stop = stopwords.words('english') + ["festival", "event", "festiv", "day", "week", "month", "year", "much"\
                                            "feature", "celebration", "celebrate", "featuring", "featurin", "include", \
                                            "weekend", "event", "featuring", "enjoy", "fest", "cotopaxi", "questival", \
                                            "around", "best", "including", "great", "first", "come", "throughout", "area", \
                                            "festivals", "events", "fairs", "days", "celebrations", "fests", "includes", \
                                            "features", "celebrating", "areas"]

        proc_text = " ".join(word for word in proc_text.split() if word not in stop)

        # Tokenizes and lemmatizes words
        proc_text = word_tokenize(proc_text)
        lemztr = WordNetLemmatizer()
        proc_text = ' '.join([lemztr.lemmatize(word) for word in proc_text])

        #self.proc_text = proc_text
        return proc_text


    def print_nb_results(self):
        # use self.nb_model & descr_text to create desired output
        # prints the naive bayes predictions

        #self.proc_text = self.proc_text.reshape(1, -1)
        #tf_idf = TfidfVectorizer(ngram_range=(1,2), min_df=10)
        #new_doc_tfidf = tf_idf.transform(self.proc_text)

        
        #count_vec = CountVectorizer(ngram_range=(1,2), min_df=10)
        #new_doc_count = count_vec.transform(new_doc_aslist)


        new_doc_aslist = [self.proc_text]
        new_doc_count = self.count_vec.transform(new_doc_aslist)


        prediction = self.nb_model.predict(new_doc_count)

        cnt_predic_list = list(prediction)
        cnt_prob_list = []
        cnt_prob_array = self.nb_model.predict_proba(new_doc_count)

        for i in range(len(cnt_prob_array)):
            cnt_prob_array.sort()

        for i in range(len(cnt_prob_array)):
            cnt_prob_list.append(cnt_prob_array[i][-1])

        cnt_label_prob = list(zip(cnt_predic_list, cnt_prob_list))

        prob_array =list(zip(self.nb_model.classes_, self.nb_model.predict_proba(new_doc_count)[0]))
        #prob_array = self.nb_model.predict_proba(new_doc_count)
        
        print("\nThe best category match is " + str(cnt_label_prob[0][0]) + " with a probability of " + str(cnt_label_prob[0][1]))
        print ("\nThe probabilities for each of the labels are " + str(prob_array))
        


    def print_lda_results(self, descr_text): # or pass processed desc text
        # use self.lda_model & descr_text to create desired output
 
        pass



    def print_d2v_results(self, descr_text):
        # use self.d2v_model & descr_text to create desired output
        pass

    def print_summary(self):
        # perhaps print everything nicely together
        # for testing
        

        if self.verbose >= 1:
            # print out the description text to analyze
            print(f"\nThe text you entered to analyze is \"{self.descr_text}\".")
        if self.verbose > 1:
            # also print out the processed text
            print(f"\nThe processed text is \"{self.proc_text}\".")
        # always print out the label
        self.print_nb_results()




if __name__ == '__main__':
    # parse argument from command line
    parser = argparse.ArgumentParser(description = 'Event description analyzer')
    parser.add_argument('--descr', '-d', help = 'Event description text')
    parser.add_argument('--verbose', '-v', help = 'Verbosity of the output: 0 for just label predictions(default); 1 for labels and the original event description; >=2 for labels, original text, and preprocessed text', default = 0)
    args = parser.parse_args()
    descr_text = args.descr
    verbose = int(args.verbose)

    # If called as a script, what to run:
    aed = AnalyzeEventDescription(descr_text, verbose)
    aed.print_summary()
    #aed.print_nb_results()

