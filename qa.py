# coding=utf-8
"""
qa.py
"""

# Library imports

# standard library
from collections import defaultdict
import re
import pickle
from google_qa import search
from stanfordcorenlp import StanfordCoreNLP


# Parameters used to score results returned from the Google-based
# system


def pretty_qa(question, num=10):
    """
    Wrapper for the `google_qa` function.
    """
    print("\nQ: " + question)
    for (j, (answer, score)) in enumerate(google_qa(question)[:num]):
        print("%s. %s (%s)" % (j + 1, answer, score))


def google_qa(question, **kwargs):
    """
    Return a list of tuples whose first entry is a candidate answer to
    `question`, and whose second entry is the score for that answer.
    The tuples are ordered in decreasing order of score.  
    """
    nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05', lang='zh')
    answer_scores = defaultdict(int)
    answer_types = [('PERSON',), ('STATE_OR_PROVINCE', 'CITY'), ('DATE', 'TIME')]
    if question.startswith('谁') or question.endswith('谁'):
        answer_type = answer_types[0]
        max_ngram = 1
    elif '哪里' in question:
        answer_type = answer_types[1]
        max_ngram = 2
    else:
        answer_type = answer_types[2]
        max_ngram = 3
    query_list = rewritten_queries(question)
    for query in query_list:
        for summary in get_summaries(query.query, **kwargs):
            for sentence in sentences(summary, nlp):
                for ngram in candidate_answers(sentence, query.query, answer_type, max_ngram):
                    answer_scores[ngram] += ngram_score(
                        ngram, query.score)
    ngrams_with_scores = sorted(answer_scores.items(),
                                key=lambda x: x[1],
                                reverse=True)
    return [("".join(ngram), score)
            for (ngram, score) in ngrams_with_scores]


def google_qa_quick(questions, **kwargs):
    """
    Return a list of tuples whose first entry is a candidate answer to
    `question`, and whose second entry is the score for that answer.
    The tuples are ordered in decreasing order of score.  
    """
    nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05', lang='zh')
    all_summary = []
    print('start qa_quick')
    try:
        for index, question in enumerate(questions):
            print('Start dealing with question {}.'.format(index))
            all_summary.append(get_summaries(question, **kwargs))
    except:
        pass

    result = []
    answer_types = [('PERSON',), ('STATE_OR_PROVINCE', 'CITY'), ('DATE', 'TIME')]
    for question, summaries in zip(questions, all_summary):
        answer_scores = defaultdict(int)
        if question.startswith('谁') or question.endswith('谁'):
            answer_type = answer_types[0]
            max_ngram = 1
        elif '哪里' in question:
            answer_type = answer_types[1]
            max_ngram = 2
        else:
            answer_type = answer_types[2]
            max_ngram = 3
        for summary in summaries:
            for sentence in sentences(summary, nlp):
                for ngram in candidate_answers(sentence, question, answer_type, max_ngram):
                    answer_scores[ngram] += ngram_score(
                        ngram, 1)
        ngrams_with_scores = sorted(answer_scores.items(),
                                    key=lambda x: x[1],
                                    reverse=True)
        result.append([("".join(ngram), score)
                       for (ngram, score) in ngrams_with_scores])
    return result


def rewritten_queries(question):
    """
    Return a list of RewrittenQuery objects, containing the search
    queries (and corresponding weighting score) generated from
    `question`.  
    """
    # rewrites = []
    # tq = tokenize(question)
    # verb = tq[1]  # 目前只处理 谁是XXX类 问题
    # rewrites.append(
    #     RewrittenQuery("\"%s %s\"" % (verb, " ".join(tq[2:])),
    #                    QUOTED_QUERY_SCORE))
    # for j in range(2, len(tq)):
    #     rewrites.append(
    #         RewrittenQuery(
    #             "\"%s %s %s\"" % (
    #                 " ".join(tq[2:j + 1]), verb, " ".join(tq[j + 1:])),
    #             QUOTED_QUERY_SCORE))
    # rewrites.append(RewrittenQuery(" ".join(tq[2:]), UNQUOTED_QUERY_SCORE))
    return [RewrittenQuery(question, 1)]


# def tokenize(question):
#     """
#     Return a list containing a tokenized form of `question`.  Works by
#     lowercasing, splitting around whitespace, and stripping all
#     non-alphanumeric characters.  
#     """
#     return list(jieba.cut(question))
#     # return nlp.word_tokenize(question)


class RewrittenQuery:
    """
    Given a question we rewrite it as a query to send to Google.
    Instances of the RewrittenQuery class are used to store these
    rewritten queries.  Instances have two attributes: the text of the
    rewritten query, which is sent to Google; and a score, indicating
    how much weight to give to the answers.  The score is used because
    some queries are much more likely to give highly relevant answers
    than others.
    """

    def __init__(self, query, score):
        self.query = query
        self.score = score


def get_summaries(query, **kwargs):
    """
    Return a list of the top 10 summaries associated to the results
    for `query` returned by `source`.  Returns all available summaries
    if there are fewer than 10 summaries available.  Note that these
    summaries are returned as BeautifulSoup.BeautifulSoup objects, and
    may need to be manipulated further to extract text, links, etc.
    Note also that we use GOOGLE_CACHE to cache old results, and will
    preferentially retrieve from the cache, whenever possible.
    """
    kwargs.update(stop=40)
    results = search(query, **kwargs)
    return results


def sentences(summary, nlp):
    """
    Return a list whose entries are the sentences in the
    BeautifulSoup.BeautifulSoup object `summary` returned from Google.
    Note that the sentences contain alphabetical and space characters
    only, and all punctuation, numbers and other special characters
    have been removed.
    """
    text = remove_spurious_words(text_of(summary))
    all_sentence = [sentence for sentence in re.split("[。，？！\n]", text) if sentence]
    all_sentence = [re.sub('[ ]+', ' ', sentence.encode('gb2312', 'ignore').decode('gb2312')).strip() for sentence in
                    all_sentence]
    return [nlp.ner(sentence) for sentence in all_sentence if sentence]


def text_of(soup):
    """
    Return the text associated to the BeautifulSoup.BeautifulSoup
    object `soup`.
    """
    return ''.join([str(x) for x in soup.findAll(text=True)])


def remove_spurious_words(text):
    """
    Return `text` with spurious words stripped.  For example, Google
    includes the word "Cached" in many search summaries, and this word
    should therefore mostly be ignored.
    """
    spurious_words = ["Cached", "Similar", '的']
    for word in spurious_words:
        text = text.replace(word, "")
    return re.sub('[.、”“:：a-zA-Z%?=()（）—「 /-]', ' ', text)


def candidate_answers(sentence, query, answer_type, max_ngram):
    """
    Return all the 1-, 2-, and 3-grams in `sentence`.  Terms appearing
    in `query` are filtered out.  Note that the n-grams are returned
    as a list of tuples.  So a 1-gram is a tuple with 1 element, a
    2-gram is a tuple with 2 elements, and so on.
    """
    filtered_sentence = [word[0] for word in sentence
                         if word[0] not in query and word[1] in answer_type]
    return sum([ngrams(filtered_sentence, j) for j in range(1, max_ngram + 1)], [])


def ngrams(words, n=1):
    """
    Return all the `n`-grams in the list `words`.  The n-grams are
    returned as a list of tuples, each tuple containing an n-gram, as
    per the description in `candidate_answers`.
    """
    return [tuple(words[j:j + n]) for j in range(len(words) - n + 1)]


def ngram_score(ngram, score):
    """
    Return the score associated to `ngram`.  The base score is
    `score`, but it's modified by a factor which is
    `CAPITALIZATION_FACTOR` to the power of the number of capitalized
    words.  This biases answers toward proper nouns.
    """
    return score + len(ngram) * 0.1


if __name__ == "__main__":
    pretty_qa('谁是中科大校长')
    pretty_qa('谁是中科大最漂亮的女生')
    pretty_qa('万立骏是哪里人')
    pretty_qa('中科大什么时候六十周年校庆')
