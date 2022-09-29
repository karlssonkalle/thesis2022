# Topic model used for thesis spring 2022
# Remember that the parameters depends on the data and the task at hand. Test and adjust accordingly.
# You need to add this manually: python -m nltk.downloader stopwords

#Packages:

#General:
import numpy as np
import pandas as pd

pd.options.display.max_columns = 100
pd.set_option('display.width', 1000)
import re, nltk, spacy, gensim
import os

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
#from gensim.utils import lemmatize  <-- Deprecated.
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

#nltk.download('stopwords')

stop_words = stopwords
stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend([
    "surveillance", "technology", "surveillance technology", "police", "http",
    "new", "use", "https", "tech", "say", "datum", "date"])

#stop_words = set(stopwords.words('english'))
#stop_words.append(['surveillance', 'technology'])

# Import Dataset
df = pd.read_csv('INPUT FILE')
#print (df.head(15))

# Convert to list
data = df.text.values.tolist() 
# Text = columnm with data


#Tokenize/clean:
def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True)
               )  # deacc=True removes punctuations


data_words = list(sent_to_words(data))
#print(data_words[:1])

# Bigrams and trigrams:
# Adjust this iteratively
bigram = gensim.models.Phrases(
    data_words, min_count=30, threshold=8.0, delimiter='_',
    scoring='default')  # higher threshold fewer phrases created.
trigram = gensim.models.Phrases(bigram[data_words],
                                min_count=30,
                                threshold=8.0,
                                delimiter='_',
                                scoring='default')
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def process_words(texts,
                  stop_words=stop_words,
                  allowed_postags=[
                      'NOUN',
                      'ADJ',
                  ]):  #Which types of words to include
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[
        word for word in simple_preprocess(str(doc)) if word not in stop_words
    ] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([
            token.lemma_ if token.lemma_ not in ['-PRON-'] else ''
            for token in doc if token.pos_ in allowed_postags
        ]))
    return texts_out


data_lemmatized = process_words(data_words, allowed_postags=['NOUN', 'ADJ'])
print(data_lemmatized)

vectorizer = CountVectorizer(
    analyzer='word',
    min_df=100,  # minimum reqd occurences of a word. Adjust accordingly!
    stop_words=stop_words,  # remove stop words
    lowercase=True,  # convert all words to lowercase
    token_pattern=
    '[a-zA-Z0-9_#]{5,}',  # Which type of tokens get accepted and minimum length (5). 
    # max_features=50000,             # max number of uniq words. Didn't use this one!
)

data_vectorized = vectorizer.fit_transform(data_lemmatized)
#Set number of topics here. For deciding learning decay and max_iter I use Gridsearch.
number_topics = 4
# Build LDA Model
# Adjust and evaluate. 
lda_model = LatentDirichletAllocation(
    n_components=number_topics,  # Number of topics
    max_iter=80,  # Max learning iterations
    learning_method='online',
    learning_decay=0.9,
    random_state=42,  # Random state
    batch_size=256,  # n docs in each learning iter
    evaluate_every=-1,  # compute perplexity every n iters, default: Don't
    n_jobs=-1,  # Use all available CPUs
)
lda_output = lda_model.fit_transform(data_vectorized)
print(lda_model)  # Model attributes

#Tests:
# Log Likelyhood:
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))

# See model parameters
pprint(lda_model.get_params())


#Grid search:
#In short: Tests different combinations of parameters.
def gridsearch():
    # Define Search Param
    # This can take a long time to run!
    search_params = {'learning_decay': [.5, .7, .9], 'max_iter': [50, 60, 70]}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(data_vectorized)

    GridSearchCV(cv=None,
                 error_score='raise',
                 estimator=LatentDirichletAllocation(batch_size=128,
                                                     doc_topic_prior=None,
                                                     evaluate_every=-1,
                                                     learning_decay=0.7,
                                                     learning_method=None,
                                                     learning_offset=10.0,
                                                     max_doc_update_iter=100,
                                                     max_iter=10,
                                                     mean_change_tol=0.001,
                                                     n_components=10,
                                                     n_jobs=1,
                                                     perp_tol=0.1,
                                                     random_state=42,
                                                     topic_word_prior=None,
                                                     total_samples=1000000.0,
                                                     verbose=0),
                 iid=True,
                 n_jobs=1,
                 param_grid={
                     'n_topics': [5, 10, 15],
                     'learning_decay': [0.5, 0.7, 0.9],
                     'max_iter': [40, 50]
                 },
                 pre_dispatch='2*n_jobs',
                 refit=True,
                 return_train_score='warn',
                 scoring=None,
                 verbose=0)
    # Best Model
    best_lda_model = model.best_estimator_
    # Model Parameters
    print("Best Model's Params: ", model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)
    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))


#gridsearch() #Here I can call the gridsearch function.


# Print out the identified topics!
def print_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join(
            [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


number_words = 10  #How many keywords to print out for each topic!
print("Topics found via LDA:")
print_topics(lda_model, vectorizer, number_words)

topic_values = lda_model.transform(data_vectorized)
topic_values.shape

# Create Document - Topic Matrix
lda_output = lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2),
                                 columns=topicnames,
                                 index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic


# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)


def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)


# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(
    color_green).applymap(make_bold)
#Function to save df to csv:
df_document_topic.to_csv('topics_text', encoding='utf-8', index=False)

#Adds topic to the dataframe
df['Topic'] = topic_values.argmax(axis=1)

print(df.head())
#Function to save df to csv:
df.to_csv('importfile_topics', encoding='utf-8', index=False)

# Topic-Keyword Matrix. Very useful for overview!
df_topic_keywords = pd.DataFrame(lda_model.components_)
# column names
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

# View
#print(df_topic_keywords.head())


def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


topic_keywords = show_topics(vectorizer=vectorizer,
                             lda_model=lda_model,
                             n_words=15)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = [
    'Word ' + str(i) for i in range(df_topic_keywords.shape[1])
]
df_topic_keywords.index = [
    'Topic ' + str(i) for i in range(df_topic_keywords.shape[0])
]
df_topic_keywords.to_csv('topic_keywords.csv', encoding='utf-8', index=False)
#f = open('NAME OF OUTPUTFILE AS TXT', 'w')
#print(df_topic_keywords, file = f)

# Build the Singular Value Decomposition(SVD) model
svd_model = TruncatedSVD(n_components=2)  # 2 components
lda_output_svd = svd_model.fit_transform(lda_output)

# X and Y axes of the plot using SVD decomposition
x = lda_output_svd[:, 0]
y = lda_output_svd[:, 1]

# Weights for the 10 columns of lda_output, for each component
print("Component's weights: \n", np.round(svd_model.components_, 2))

# Percentage of total information in 'lda_output' explained by the two components
print("Perc of Variance Explained: \n",
      np.round(svd_model.explained_variance_ratio_, 2))

# Plot - uncomment if you want to plot
#plt.figure(figsize=(12, 12))
#plt.scatter(x, y, c=clusters, cmap=plt.cm.get_cmap('tab10', 10))
#plt.xlabel('Component 2')
#plt.xlabel('Component 1')
#plt.title("Segregation of Topic Clusters", )
# plot the results
#plt.colorbar(ticks=range(10), label='digit value')
#plt.clim(-0.5, 9.5)
#plt.show() #Uncomment if you want to plot this!

#Visualize with LDAvis
LDAvis_data_filepath = os.path.join('./ldavis_prepared_' + str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda_model, data_vectorized,
                                          vectorizer)
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
# It gets saved as a file so you can view it later as well.
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared,
                   './ldavis_prepared_' + str(number_topics) + '.html')
pyLDAvis.display(LDAvis_prepared)
