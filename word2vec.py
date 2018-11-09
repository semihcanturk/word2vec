import requests
import json
from bs4 import BeautifulSoup
from gensim import corpora, models
from scipy import spatial
import gensim
import re
import nltk
import pickle
import os.path
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')


def build_tfidf_dict(corpus_tfidf, steps_dict):
    """
    Computes tf-idf coefficients for each word in each document
    @param corpus_tfidf: list of list of tuples, a mapping of integers for each word to their tf-idf coefficient
     in each document
    @param steps_dict: dict, a mapping of words to integers
    @return: tfidf_vals, tf-idf coefficients for each word in each document
    """
    tfidf_vals = []
    for doc in corpus_tfidf:
        dict = {}
        for id, value in doc:
            word = steps_dict.get(id)
            dict[word] = value
        tfidf_vals.append(dict)
    return tfidf_vals


def vectorize(text):
    """
    Vectorizes the steps based on its match with a given text by tf-idf adjusting and averaging matching words.
    @param text: a bag-of-words on whose matches with the steps will be tf-idf adjusted and averaged to produce a
    single vector for each step
    @return: vlist, the list of lists which has a vector for each step representing its similarity
    """
    step_count = 0
    vlist = []
    for step in steps_parsed:
        step_count = step_count + 1
        d = tfidf_vals[step_count - 1]  #get the tf-idf coefficients for the step
        ctr = 0
        query_vect = []
        for word in text:
            if word not in stops and word in step:  # for each word in the text, we consider it only if it is found in
                if len(query_vect) == 0:            # the current step, if not we skip over
                    query_vect = word_vectors[word] * d[word]
                    ctr = ctr + 1
                else:
                    query_vect = query_vect + word_vectors[word] * d[word]
                    ctr = ctr + 1
        query_vect[:] = [x / ctr for x in query_vect]   # average the vectors
        vlist.append(query_vect)
    return vlist


def vectorize_steps(steps_parsed, tfidf_vals, word_vectors):
    """
    For each step, gets vector representations for each word, scales them with the tf-idf coefficients,
    and averages them. As a result, each step is represented by a single 300-length vector. This is a slightly
    edited version of vectorize() that is used to vectorize the steps themselves without regard to an outside text
    @param steps_parsed: a bag-of-words for each step
    @param tfidf_vals: tf-idf coefficients for each word in each document
    @param word_vectors: vector representation of each word from each document
    @return: vectors, averaged word vectors adjusted with their tf-idf coefficients for each document
    """
    vectors = []
    step_count = 0
    for step in steps_parsed:
        step_count = step_count + 1
        vect = []
        ctr = 0
        d = tfidf_vals[step_count - 1]
        for word in step:
            if len(vect) == 0:
                vect = word_vectors[word] * d[word]
                ctr = ctr + 1
            else:
                vect = vect + (word_vectors[word] * d[word])
                ctr = ctr + 1
        vectors.append(vect/ctr)
    return vectors


def parse_steps(steps, word_vectors, stops, regex, sep):
    """
    Parses the text and saves each step in a list of lists.
    @param steps:   list of dicts, each dict representing a step
    @param word_vectors:    loaded word2vec vectors for each word
    @param stops:   list of ignored words
    @param regex:   chars to be cleaned
    @param sep:     list of punctuation to be cleaned
    @return: parsed steps
    """
    steps_parsed = []
    for i in steps:
        w = []
        step = i["itemListElement"]["text"]
        for word in step.lower().split():
            if word not in stops and regex.search(word) is None:    # ignore words with unrecognized chars
                if sep.search(word) is not None:
                    word = re.sub(r'[^\w]', '', word)   # remove non-alphanumeric char (punctuation)
                    word = word.rstrip(string.digits)
                if word in word_vectors.vocab:          # if word is in vocabulary, add it to the list of words
                    w.append(word)
        steps_parsed.append(w)
    return steps_parsed


def load_model():
    """
    Extracts the text from the appropriate WikiHow article or loads it if it is already extracted and saved.
    Extracts and trains the Google Word2Vec model or loads it if it is already extracted and saved.
    Defines regex, punctuation, stopwords (commonly occurring words in English), common_words (words that are
    common in the given article) to strip from text.
    Defines a SnowballStemmer to stem words.
    Defines a dict num_to_step to map the array of text sections to the appropriate methods and steps
    @return: steps_parsed, tfidf_vals, vectors, word_vectors, num_to_step, regex, sep, st, stops, common_words
    """
    if os.path.isfile('steps.pckl'):    # load saved article
        f = open('steps.pckl', 'rb')
        steps = pickle.load(f)
        f.close()
    else:   # or fetch from address
        page = requests.get("https://www.wikihow.com/Fry-an-Egg")
        soup = BeautifulSoup(page.content, "html.parser")
        scripts = soup.find_all('script', type='application/ld+json')
        s1 = scripts[1]
        body = s1.contents[0]
        j = json.loads(body)

        method_1 = j["step"][0]
        steps_1 = method_1["itemListElement"]
        method_2 = j["step"][1]
        steps_2 = method_2["itemListElement"]
        steps = steps_1 + steps_2   # merge steps from both methods

        f = open('steps.pckl', 'wb')
        pickle.dump(steps, f)
        f.close()

    regex = re.compile('[@_!#$%^&*()<>?/|}{~:]')
    sep = re.compile('[.,!;?]')

    st = SnowballStemmer("english") # alternatively a more aggressive PorterStemmer could be used
    stops = stopwords.words("english")
    common_words = set("put much cook into".split())    # these are words specific to the document which we can't
    num_to_step = {0: "Method 1, Step 1",               # attribute to a few steps only, so we disregard them
                   1: "Method 1, Step 2",
                   2: "Method 1, Step 3",
                   3: "Method 1, Step 4",
                   4: "Method 1, Step 5",
                   5: "Method 1, Step 6",
                   6: "Method 1, Step 7",
                   7: "Method 1, Step 8",
                   8: "Method 2, Step 1",
                   9: "Method 2, Step 2",
                   10: "Method 2, Step 3",
                   11: "Method 2, Step 4",
                   12: "Method 2, Step 5",
                   13: "Method 2, Step 6",
                   }

    if os.path.isfile('vectors.kv'):    # load keyedvectors of the Word2Vec model
        word_vectors = gensim.models.KeyedVectors.load("vectors.kv", mmap='r')
    else:                               # or fetch and train
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                       binary=True)
        word_vectors.save("vectors.kv")

    steps_parsed = parse_steps(steps, word_vectors, stops, regex, sep)  # parse the steps in each method
    steps_dict = corpora.Dictionary(steps_parsed)                       # load them to a dictionary
    steps_corpus = [steps_dict.doc2bow(text) for text in steps_parsed]  # create a corpus out of the dictionary
    tfidf = models.TfidfModel(steps_corpus, normalize=True)             # train a tf-idf model on the corpus
    corpus_tfidf = tfidf[steps_corpus]                                  # apply tf-idf to the whole corpus
    tfidf_vals = build_tfidf_dict(corpus_tfidf, steps_dict)             # and build a dict of values

    vectors = vectorize_steps(steps_parsed, tfidf_vals, word_vectors)   # create vectors for each step

    return steps_parsed, tfidf_vals, vectors, word_vectors, num_to_step, regex, sep, st, stops, common_words


def clean_query(query):
    """
    Parses and cleans the input query, looks for keywords to detect types of question (time, amount etc. enquiry) and
    adds new words that can potentially be found in the step that contains the answer. Also adds variations or synonyms
    of certain words to the list of keywords to search for. Finally, checks if the question is well-formed input-wise
    and ends with a question mark. If not, declares it a non-question and prompts the user to a new question.
    @param query: the input question asked by the user
    @return: query_cleaned, the query cleaned from punctuation and new keywords added according to criteria
             query_org, query cleaned minus the additional keywords
             query_split, the query lowercased and parsed without any additional processing
    """
    query_cleaned = []
    query_org = []
    query_split = query.lower().split()

    if query_split[0] == "how":
        if query_split[1] == "long":    # a "how long" query for food prep is likely to include the terms below
            query_cleaned.append("takes")
            query_cleaned.append("minutes")
            query_cleaned.append("seconds")
        if query_split[1] == "much":    # a "how much" query is likely to include the terms below in the specified doc
            query_cleaned.append("some")
            query_cleaned.append("more")
            query_cleaned.append("plenty")
            query_cleaned.append("ml")
            query_cleaned.append("tablespoons")
    if query_split[0] == "when":    # a "when" query is also likely to include "when" in the response
        query_cleaned.append("when")

    last = query_split[-1]
    if '?' not in last:     # check for question mark
        print("not a valid question")
        if writefile:
            f = open("log.txt", "a")
            f.write("not a valid question\n")
            f.close()
        start()
    else:
        query_split[-1] = last.replace('?', '')     # clean the question mark once it is found

    for word in query_split:
        if word not in stops and word not in common_words:
            # rules dependent on document, for the "How to Fry an Egg?" article, these were found to be common parallels
            # more rules can be added or altered
            if word == "egg":
                query_cleaned.append("eggs")
            if word == "crack":
                query_cleaned.append("break")
            query_cleaned.append(word)
            query_org.append(word)
            w2 = st.stem(word)
            if word != w2:
                query_cleaned.append(w2)

    return query_cleaned, query_org, query_split


def shortcut_sims(query_org):
    """
    Removes any added keywords and computes similarities of the single word with the step vectors. If there is any
    similarity (nonzero after flooring) between the two, then we can assume the step involves the keyword and hence
    is relevant.
    @param query_org: original query, single word
    @return: sims, similarities between the single word and the steps in list form
    """
    ws = ['takes', 'minutes', 'seconds', 'some', "more", "plenty", "ml", "tablespoons"]
    alt_query = query_org.copy()
    for x in query_org:
        if x in ws:
            alt_query.remove(x)
    alt_vect_list = vectorize(alt_query)

    sims = []
    i = 0
    for vect in vectors:
        a = vect
        b = alt_vect_list[i]
        if len(b) == 0:
            sims.append(0)
        else:
            sim = 1 - spatial.distance.cosine(a, b)
            sims.append(sim)
        i = i + 1
    if sum(sims) == 0:
        print("not contained")
        if writefile:
            f = open("log.txt", "a")
            f.write("not contained\n")
            f.close()
        start()
    else:
        return sims


def post_process(query_org, query_cleaned, keyword_thresh=5):
    """
    Makes rule-based altercations on the list of keywords. If the cleaned query has >= keyword_thresh keywords,
    we remove the common words "egg", "eggs" and "pan" since they add noise and given the abundance of the keywords,
    are not necessary. However, if we have < keyword_thresh keywords, then we are better off not removing these terms
    since they are then likely to be elemental to the question (unlike the global common_words variable).

    If the original query has only one keyword, then this function calls remove_terms() to shortcut the process since
    the single keyword is very likely to be contained in the answer, and only looks for the steps that have a
    correlation with the keyword.

    @param query_org: original query
    @param query_cleaned: cleaned query as an output of clean_query()
    @param keyword_thresh: defaulted at 5, the limit on which we apply common word cleaning
    @return: query_org and query_cleaned, the altered versions of the parameters
    """
    if len(query_cleaned) >= keyword_thresh:
        items = ["egg", "eggs", "pan"]
        for item in items:
            if item in query_cleaned:
                query_cleaned.remove(item)
            if item in query_org:
                query_org.remove(item)

    if len(query_org) == 1:
        sims = shortcut_sims(query_org)  # if only one keyword, then remove any added terms and compute similarity
        ctr = 0                          # with the single word
        for i in sims:
            if i != 0:
                print(num_to_step[ctr])
                if writefile:
                    f = open("log.txt", "a")
                    f.write(num_to_step[ctr] + "\n")
                    f.close()
            ctr = ctr + 1
        start()
    return query_cleaned, query_org


def compute_sims(query_vectors):
    """
    Given query_vectors, a list of vectors that result from tuning the query words to the tf-idf scores that differ
    in each step, computes the similarity of each query vector with its corresponding step vector.
    @param query_vectors: a list of vectors based on the given query, each vector is tf-idf tuned for a specific step
    @return: sims, a list of similarity scores; and norm, max_normalized scores
    """
    sims = []
    i = 0
    for vect in vectors:
        a = vect    # this is the vector representing a specific step, comes from vectorize_steps()
        b = query_vectors[i]    # this is the vector representing the query tuned to the given step
        if len(b) == 0:         # if b is empty, that means there were no matching words and hence no vector representing
            sims.append(0)      # the query, so we assume 0 similarity
        else:
            sim = 1 - spatial.distance.cosine(a, b)     # else compute cosine distance to measure similarity
            sims.append(sim)
        i = i + 1
    m = max(sims)
    if m == 0:     # if all similarity values are zero, then the answer to the question is not contained
        print("not contained")
        if writefile:
            f = open("log.txt", "a")
            f.write("not contained\n")
            f.close()
        start()
    else:
        norm = [float(i) / m for i in sims]     # normalize the similarity values for interpretability
        return sims, norm


def evaluate(norm):
    """
    For each step, if similarity with the question is above the threshold, print the step as output
    @param norm: normalized similarity vector, one score for each step
    @return: the list of steps that answer the input query
    """
    steplist = []
    for i in norm:
        if i > threshold:
            steplist.append(num_to_step[norm.index(i)])
    return steplist


def process_query(query):
    """
    Takes the input question, cleans and processes it, calculates similarity scores and outputs them. Also, double
    checks whether the "how" questions have any matches after removing added words. "How" queries tend to involve a lot
    of adjectives, so we make sure whether the rest of the query is still relevant after we remove phrases like "much"
    and "long", which happens to be crucial in catching unrelated questions.
    @param query: user input, a well-formed question regarding "How to Fry an Egg?"
    @return: the list of steps that answer the input query
    """
    query_cleaned, query_org, query_split = clean_query(query)
    query_cleaned, query_org = post_process(query_org, query_cleaned, keyword_thresh)

    if query_split[0] == "how":
        shortcut_sims(query_cleaned)    # the shortcut function removes added keywords before checking similarities
    query_vectors = vectorize(query_cleaned)
    sims, norm = compute_sims(query_vectors)
    steplist = evaluate(norm)
    return steplist


def start():
    """
    Prompts the user to enter a query and processes it, and calls start() again afterwards.
    """
    prompt = "Please enter a query on 'How to Fry an Egg?', or enter 'exit' to exit program\n"
    if writefile:
        f = open("log.txt", "a")
        f.write(prompt)
        f.close()
    query = input(prompt)
    if writefile:
        f = open("log.txt", "a")
        f.write("User Input: " + query + "\n")
        f.close()
    if query == "exit":
        exit(0)
    else:
        steplist = process_query(query)
        if len(steplist) > 0:
            for step in steplist:
                print(step)
                if writefile:
                    f = open("log.txt", "a")
                    f.write(step + "\n")
                    f.close()
        start()


if __name__ == '__main__':
    # these variables are defined global as they are accessed by most functions
    steps_parsed, tfidf_vals, vectors, word_vectors, num_to_step, regex, sep, st, stops, common_words = load_model()
    threshold = 0.62
    keyword_thresh = 5
    writefile = False   # change to True if we want to log I/O
    f = open("log.txt", "a")
    start()
