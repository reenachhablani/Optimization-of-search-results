import re
import argparse
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen
from urllib.error import HTTPError
from collections import Counter, defaultdict
from math import log10
from bs4 import BeautifulSoup
import numpy as np

prob = 0.05
target_delta = 0.04

stop_words = [
    'a', 'also', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'do',
    'for', 'have', 'is', 'in', 'it', 'of', 'or', 'see', 'so',
    'that', 'the', 'this', 'to', 'we'
]

#Taking the input
#Search query and list of urls
def input():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'query',
        type=str,
        nargs='+',
        help='Search query string can contain multiple words'
    )
    parser.add_argument(
        'url',
        type=str,
        nargs='+',
        help='At least one seed url for the crawler to start from'
    )
    return parser.parse_args()

def crawler(urls, _frontier={}, _bases=None):
    '''
    Takes a list of urls as argument and crawls them recursively until
    no new url can be found.
    Returns a sorted list of tuples (url, content, links).
    `links` is a list of urls.
    '''
    if not _bases:
        _bases = [urlparse(u).netloc for u in urls]
    for url in [u.rstrip('/') for u in urls]:
        if url in _frontier:
            continue
        try:
            response = urlopen(url)
        except HTTPError as e:
            print(e, url)
            continue

        page = parse(response, url, _bases)
        print('crawled %s with %s links' % (url, len(page[2])))
        _frontier[url] = page
        crawler(page[2], _frontier, _bases)
    return sorted(_frontier.values())


def parse(html, url, bases):
    '''
    Takes an html string and a url as arguments.
    Returns a tuple (url, content, links) parsed from the html.
    '''
    try:
        data = BeautifulSoup(html, 'lxml')

        content = data.body.get_text().strip()

        links = [urljoin(url, l.get('href')) for l in data.findAll('a')]
        links = [l for l in links if urlparse(l).netloc in bases]
    except AttributeError:
        pass

    return url, content, links

def page_rank(pages):
    '''
    Returns a matrix with documents as columns
    and values for each round as rows.
    Number of rows depends on how long it takes to reach the target_delta.
    '''
    N = len(pages)
    transition_matrix = create_transition_matrix(pages)
    ranks_in_steps = [[1 / N] * N]
    while True:
        possibilities = ranks_in_steps[-1] * transition_matrix
        delta = get_delta(possibilities, ranks_in_steps[-1])
        ranks_in_steps.append(np.squeeze(np.asarray(possibilities)))
        if delta <= target_delta:
            return ranks_in_steps

def create_transition_matrix(pages):
    '''
    Returns a matrix with document urls as rows
    and document links as columns.
    Each cell contains the probability for a document
    to transition to a link.
    '''
    links = get_links(pages)
    urls = get_urls(pages)
    N = len(pages)
    m = np.matrix([[weight_link(N, u, l) for u in urls] for l in links])
    return prob_to_transition(N, m)


def weight_link(N, url, links):
    if not links:
        return 1 / N
    if url in links:
        return 1 / len(links)
    else:
        return 0


def prob_to_transition(N, m):
    return m * (1 - prob) + prob / N


def get_delta(a, b):
    return np.abs(a - b).sum()




def get_links(pages):
    return [links for url, content, links in pages]


def best_rank(ranks, pages):
    '''
    Returns a dict with document urls as keys
    and their ranks as values.
    '''
    list_of_url = [url for url, content , links in pages]
    return dict(zip(list_of_url, ranks[-1]))

# Index

def create_index(pages):
    '''
    Returns the index as a dict with terms as keys
    and lists tuples(url, count) as values.
    Count says how many times the term occured in the document.
    '''
    index = defaultdict(list)
    for url, content, links in pages:
        counts = count_terms(content)
        for term, count in counts.items():
            index[term].append((url, count))
    return index


def count_terms(content):
    '''
    content is a text string.
    Returns a Counter with terms as keys
    and their occurence as values.
    '''
    return Counter(get_terms(content))


normalize = re.compile('[^a-z0-9]+')


def get_terms(s):
    '''
    Get a list of terms from a string.
    Terms are lower case and all special characters are removed.
    '''
    normalized = [normalize.sub('', t.lower()) for t in s.split()]
    return [t for t in normalized if t not in stop_words]


def weight_index(index, N):
    '''
    Takes an index as first argument
    and the total number of documents as second argument.
    Returns a new index with tf_idf weights instead of simple counts.
    '''
    weighted_index = defaultdict(list)
    for term, docs in index.items():
        df = len(docs)
        for url, count in docs:
            weight = tf_idf(count, N, df)
            weighted_index[term].append((url, weight))
    return weighted_index


def tf_idf(tf, N, df):
    return 1 + log10(tf) * log10(N / df)

#word term frequency -  1 + log10(tf)
#in document frequency -  log10(N / df)


def normalize_index(index):
    '''
    Takes an index as argument.
    Returns a new index with normalized weights.
    '''
    lengths = doc_lengths(index)
    norm_index = defaultdict(list)
    for term, docs in index.items():
        for url, weight in docs:
            norm_index[term].append((url, weight / lengths[url]))
    return norm_index


def doc_lengths(index):
    '''
    Returns a dict with document urls as keys
    and vector lengths as values.
    The length is calculated using the vector of weights
    for the terms in the document.
    '''
    doc_vectors = defaultdict(list)
    for docs in index.values():
        for url, weight in docs:
            doc_vectors[url].append(weight)
    return {url: np.linalg.norm(doc) for url, doc in doc_vectors.items()}


# Search & Scoring

def cosine_similarity(index, N, query):
    '''
    query is a string of terms.
    Returns a sorted list of tuples (url, score).
    Score is calculated using the cosine distance
    between document and query.
    '''
    scores = defaultdict(int)
    print(query)
    terms = query[0]
    qw = {t: tf_idf(1, N, len(index[t])) for t in terms if t in index}
    query_len = np.linalg.norm(list(qw.values()))
    for term in qw:
        query_weight = qw[term] / query_len
        for url, weight in index[term]:
            scores[url] += weight * query_weight
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def combined_search(index, N, rank, query):
    '''
    Returns a sorted list of tuples (url, score).
    Score is the product of the cosine similarity and the PageRank.
    '''
    scores = cosine_similarity(index, N, query)
    combined = [(doc, score * rank[doc]) for doc, score in scores]
    return sorted(combined, key=lambda x: x[1], reverse=True)


def print_combined_search(index, N, rank, query):
    print('Search results for "%s":' % (query))
    for url, score in combined_search(index, N, rank, query):
        print('%.6f  %s' % (score, url))


def main():
    args = input()
    # Computing
    pages = crawler(args.url)
    ranks = page_rank(pages)
    rank = best_rank(ranks, pages)
    N = len(pages)
    index = create_index(pages)
    weighted_index = weight_index(index, N)
    norm_index = normalize_index(weighted_index)

    # Print results
    print()
    print('Number of pages:', len(pages))
    print('Terms in index:', len(index))
    print('Iterations for PageRank:', len(ranks))
    print()
    print_combined_search(norm_index, N, rank, args.query)




if __name__ == "__main__":
    main()
