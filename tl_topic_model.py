'''
This code accompanies an article in Theological Librarianship called Topic Modeling: An Introduction and Narrative.
It is written for pedagogical purposes, not to be as efficient as possible. Not included here is the script which scraped
the website of Theological Librarianship; the result of that scraping was a list of Python dictionaries. Each dictionary
represented an article and was structured like this:
{'authors': 'Sandra Oslund',
 'id': '4331518',
 'issue': 'Vol 9 No 2 (2016)',
 'text': '9PROFILES: Quiet Person, Powerfully Loud In˜uence: Norris Magnuson (1932-2006)by Sandra OslundNorris Alden
          Magnuson was born to George and Esther (Eliason) Magnuson on \n...'
'title': 'Quiet Person, Powerfully Loud Influence: Norris Magnuson '
          '(1932-2006)'}
'''
# import necessary Python packages
import json # used to save/load json files
import pickle # used to save/load pickle files
import spacy # used for text pre-processing
import re # used to find patters for text preporcessing
from gensim import corpora, models # used for initializing the topic model
from pprint import pprint


#load the dictionary containing metadata and raw text extracted from pdfs
tl_corpus = pickle.load(open('tl_corpus.pickle', 'rb'))


'''
Pre-processing: The following lines of code create a Python dictionary which maps each document index number to the 
article id. This will later be used to match documents from the topic model to the correct article metadata. In addition,
the following lines begin pre-processing by cleaning up the text extracted from the pdfs.
'''
docs = []
doc_index2article_id = {}
i = 0
for article_dict in tl_corpus:
    id = article_dict["id"]
    doc_index2article_id[i] = id
    i += 1
    raw_text = article_dict["text"]
    text_lower = raw_text.lower() # normalize by setting all letters to lowercase
    text_no_punct = re.sub(r"[.,<>/?';:@ł™#$%&*+\(\)\-\d\"\[\]]", " ", text_lower) # remove punctuation
    text_no_symbols = re.sub(r"˛", "ff", text_no_punct)  # clean up text from pdf
    text_no_symbols = re.sub(r"˜", "th", text_no_symbols)  # clean up text from pdf
    text_no_symbols = re.sub(r"˚", "fi", text_no_symbols)  # clean up text from pdf
    docs.append(text_no_symbols)

# save mappinng dictionary for later use
with open('doc_index2article_id.json', mode='w') as f:
    json.dump(doc_index2article_id, f)

# more preprocessing: remove stop words, numbers, etc
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
processed_docs = []
for doc in nlp.pipe(docs):
    doc = [token.lemma_ for token in doc if token.is_alpha]
    doc = [token for token in doc if token not in stop_words]
    doc = [token for token in doc if len(token) > 2]
    processed_docs.append(doc)

'''
Processing: At this point we have a list of preprocessed documents. Each preprocessed document itself is a list of 
lemmatized tokens. Now these documents can be processed into a corpus which Gensim can use to train a topic model.
'''

dictionary = corpora.Dictionary(processed_docs)  # create Gensim dictionary which maps word ids to word counts
dictionary.filter_extremes(no_below=5, no_above=0.5)  # filter out words which are too frequent or too rare
dictionary.save('tl_corpus.dict')  # save for later use
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]  # initialize Gensim corpus
corpora.MmCorpus.serialize('corpus.mm', corpus)  # save for later use

'''
Initiate the topic model. Here I generated 4 different models in which I changed the num_topics parameter to see which
generated the most informative results.
'''
lda25 = models.LdaModel(corpus, id2word=dictionary, num_topics=25, passes=100)  # initiate model
lda25.save('lda25.model')  # save for later use

lda50 = models.LdaModel(corpus, id2word=dictionary, num_topics=50, passes=100)  # initiate model
lda50.save('lda50.model')  # save for later use

lda75 = models.LdaModel(corpus, id2word=dictionary, num_topics=75, passes=100)  # initiate model
lda75.save('lda75.model')  # save for later use

lda100 = models.LdaModel(corpus, id2word=dictionary, num_topics=100, passes=100)  # initiate model
lda100.save('lda100.model')  # save for later use

'''
Now that the topic model is complete, it can be analyzed
'''
# Inspect the words in each topic
pprint(lda50.show_topics(num_topics=50))

# Inspect documents associated with a topic
i = 0
for doc in corpus:
    topics = lda50.get_document_topics(doc, minimum_probability=0.3)  #only get docs with a 30% probability of association
    print('doc ', i, '=', topics)
    i += 1

# find associated metadata with each document in a given topic (here topic 39)
doc_ids = [doc_index2article_id['186'], doc_index2article_id['187'],doc_index2article_id['188'],
                   doc_index2article_id['189'],doc_index2article_id['190'], doc_index2article_id['191']]
for item in doc_ids:
    for article in tl_corpus:
        if item == article['id']:
            print(article['authors'], article['title'], article['issue'])
