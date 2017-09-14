import nltk
from nltk.tokenize import TweetTokenizer
import string
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
from bokeh.plotting import figure
from bokeh.charts import Bar, Scatter, output_file, show
import SocialDash as soda
import sys

#need to build tokenizer and then can iterate through the dataframe as follows

#### 0. Utility fucntion to flatten nested lists or tuples into a single list
def flatten_list(token_list):
	"""takes the list of lists returned by ngrams and returns a single list of strings"""
	output = [item for sublist in token_list for item in sublist]
	return output

##### 1. break a vector of text data into lists of words

def make_tokens(tweet_vec):
	"""Tokenize a vector of tweets from a pandas data frame. Must specifiy
	a data frame column df["tweets"]. Returns a list of lists containing 
	lowercase tokens"""

	tt = TweetTokenizer()
	tokens = []

	for line in tweet_vec:
		token = tt.tokenize(line.lower())				
		tokens.append(token)
	return tokens

#### 2. Remove punctuation and stopwords from lists of tokens
def punctuation_filter(tokens):
	"""Removes punctuation marks from the posts. Also removes emoticons.
	this function retains hashtags, links, and @ mentions."""
	filtered_tok = [[punc for punc in token if punc not in string.punctuation]for token in tokens]
	return filtered_tok

def stopword_filter(tokens):
	"""Removes stopwords from the tweets via the nltk.corpus.stopwords.words 
	dictionary."""
	filtered_tok = [[word for word in token if word not in stopwords.words('english')] for token in tokens]
	return filtered_tok

def remove_punc_stop(tokens):
	"""Returns a clean list with no stopwords or punctuation marks"""
	clean = stopword_filter(punctuation_filter(tokens))
	clean = flatten_list(clean)
	return clean

##### 3. Separate out the hashtags, at mentions, and words

def tag_extract(flat_grams):
	"""Extract hashtags from list of most common words. Should only feed in 
	unigrams"""
	if 'hashtags' not in locals():
		hashtags = []
	if 'at_mentions' not in locals():
		at_mentions = []
	for word in flat_grams:
		if word[0] == '#':
			hashtags.append(word)
		if word[0] == '@':
			at_mentions.append(word)
	return hashtags, at_mentions


def words_no_tags(flat_grams, remove_at = True):
	"""Extract only words (no hashtags) from the unigram list.
	For some reason this method leaves in empty values in the list. 
	Will have to check to make sure it doesn't screw anything up. 
	Can separate @ mentions or leave them in"""
	notags = [word for word in flat_grams if word[0] != '#']
	if remove_at == True:
		notags = [word for word in notags if word[0] != '@']
	return notags


#### 4. Read through list of lists of toekns to create ngrams of size n
def ngrams(tokens, n):
	"""Reads in a list of lists of tokens of tweets (so a list of tokens from tweets)
	and iterates through to create ngrams of size n"""
	nlist = [grams for grams in nltk.ngrams(tokens, n)]
	return nlist

#### 5. Count the frequency of each ngram

def ngram_count(flat_list, n):
	"""Returns a sorted frequency distribution and most common n words."""
	fdist = nltk.FreqDist(flat_list)
	n_common = nltk.FreqDist.most_common(fdist,n)
	return n_common



def clean_for_plot(counts, ngrams = 1, tags_at = False):
	"""Reads in the list from ngram_count anmd outpiuts a Pandas
	DataFame with words as the key and frequencies as the values
	Options: ngrams - how long of grams to expect (1 -unigrams, 2-bigrams)
	"""
	names_1, counts_1 = zip(*counts)
	#stripnames = punctuation_filter(names_1)
	if tags_at == False:
		f_names = flatten_list(names_1)
		f_names = [x.capitalize() for x in f_names]
	else:
		f_names = names_1
	f_names2 = [] 
	l_counts = list(counts_1)
	dictionary = dict(zip(f_names, l_counts))
	if ngrams == 2:
		for i in range(0,len(f_names),2):
			f_names2.append(f_names[i] + ", " + f_names[(i+1)])
		dictionary = dict(zip(f_names2, l_counts))
	if ngrams == 3:
		for i in range(0,len(f_names),3):
			f_names2.append(f_names[i] + ", " + f_names[(i+1)] + ", " +f_names[(i+2)])
		dictionary = dict(zip(f_names2, l_counts))	
	df = pd.DataFrame.from_dict(dictionary, orient = 'index')
	df.columns = ['Freq']
	df = df.sort_values(by = 'Freq', ascending = False)
	return df

def plot_grams_freq(df, title = "title", color = "orange"):
	"""Make a horizontal bar chart of the most frequently used words in a tweet"""
	
	figname = figure(x_axis_label = "Word Frequency", y_axis_label = "Word",
		y_range = df.index.tolist(), title = title)
	figname.hbar(y = df.index.tolist(), height = .5, left = 0, right = df["Freq"],
		fill_color = color, line_color = color)
	#show(auth_fig)
	#print(df)
	return figname


if __name__ == "__main__":

	from SocialDash import nuvi

	f = sys.argv[1]
	n = int(sys.argv[2])
	output = sys.argv[3]
	nuvi_dat = False

	dat = pd.read_csv(f)
	dat_small, dat_eng, twit, nuvi_dat = nuvi(data = dat)

	toks = make_tokens(twit["Text"])
	toks_clean = remove_punc_stop(toks)
	tags, ats = tw.tag_extract(toks_clean)
	words = words_no_tags(toks_clean)

	unigrams = ngrams(words,1)
	bigrams = ngrams(words,2)
	trigrams = ngrams(words,3)

	###Counts
	uni_count = ngram_count(unigrams, n)
	bi_count = ngram_count(bigrams, n)
	tri_count = ngram_count(trigrams, n)
	tag_count = ngram_count(tags, n)
	at_count = ngram_count(ats, n)

	###Create DataFrames for Plots
	uni_clean = clean_for_plot(uni_count, ngrams = 1, tags_at = False)
	bi_clean = clean_for_plot(bi_count, ngrams = 2, tags_at = False)
	tri_clean = clean_for_plot(tri_count, ngrams = 3, tags_at = False)
	tag_clean = clean_for_plot(tag_count, ngrams = 1, tags_at = True)
	at_clean = clean_for_plot(at_count, ngrams = 1, tags_at = True)

	###Set up plots- Horizontal bars
	t1 = plot_grams_freq(uni_clean, title = "Most Frequent Unigrams")
	t2 = plot_grams_freq(bi_clean, title = "Most Frequent Bigrams")
	t3 = plot_grams_freq(tri_clean, title = "Most Frequent Trigrams")
	t4 = plot_grams_freq(at_clean, title = "Most Frequently Mentioned Users")
	t5 = plot_grams_freq(tag_clean, title = "Most Frequently Used Hashtags")
	t6 = soda.plot_nets(twit)
	t7 = soda.plot_count_posts(twit, x_label = "Number of Posts", title = "Highest Posting Users")
	###Create Tabs for plots
	p2 = row(column(t1,t6), t2)
	p3 = column(row(t4,t7), t5)

	tab1 = Panel(child = p2, title = "Netowrk Use and Word Frequency")
	tab2 = Panel(child = p3, title = "Users and Hashtags")

