import argparse
from collections import defaultdict
import doc_store as ds
import json
import math
import process_wiki as pw
import spacy
import time
from doc_store import DocStore, InMemoryDocStore
from inverted_index import InvertedIndex
from tqdm import tqdm


class BigramModel:
    def __init__(self, nlp, doc_store, bow_store, inverted_index, vocab, doc_freqs):
        assert nlp is not None
        assert doc_store is not None
        assert bow_store is not None
        assert isinstance(vocab, dict)
        assert isinstance(doc_freqs, dict)
        assert 'TOTAL' in doc_freqs
        self.nlp = nlp
        self.doc_store = doc_store
        self.bow_store = bow_store
        self.inverted_index = inverted_index
        self.vocab = vocab
        self.doc_freqs = doc_freqs
        self.total_docs = doc_freqs['TOTAL']
        self.unknown_id = len(vocab)

    def search(self, query, result_limit, inverted_index_limit, add_snippet=False,
               remove_stopwords=False, debug=False):
        assert isinstance(query, str)
        assert result_limit > 0
        assert inverted_index_limit > 0

        timings = {}

        start = time.time()
        query_bow = self.create_bow(query, remove_stopwords=remove_stopwords)
        timings['query_processing'] = time.time() - start
        if debug:
            print('Query_bow={}'.format(query_bow))

        start = time.time()
        doc_ids = self.inverted_index.get(bow=query_bow, limit=inverted_index_limit)
        timings['inverted_index'] = time.time() - start
        if debug:
            print('Retrieved {} doc_ids from inverted index, example: {}'.format(len(doc_ids), doc_ids[:5]))

        start = time.time()
        doc_bows = self.bow_store.get(doc_ids)
        doc_bows = [x['bow'] for x in doc_bows]
        timings['bow_retrieval'] = time.time() - start

        start = time.time()
        doc_scores = [self.score(doc_bow=doc_bow, query_bow=query_bow) for doc_bow in doc_bows]
        timings['scoring'] = time.time() - start

        start = time.time()
        idxs = list(range(len(doc_scores)))
        idxs = sorted(idxs, key=lambda x: -doc_scores[x])[:result_limit]
        top_ids = [doc_ids[i] for i in idxs]
        top_scores = [doc_scores[i] for i in idxs]
        search_results = [{'id': id, 'score': score} for id, score in zip(top_ids, top_scores)]
        timings['sorting'] = time.time() - start

        if add_snippet:
            start = time.time()
            top_docs = self.doc_store.get(top_ids)
            for i in range(len(search_results)):
                search_results[i]['doc'] = top_docs[i]
            timings['snippet'] = time.time() - start

        if debug:
            print('\nTimings:')
            print('-' * 40)
            print('{:<20}{}'.format('step', 'duration (sec)'))
            print('-' * 40)
            for step, duration in timings.items():
                print('{:<20}{:0.2f}'.format(step, duration))
            print('\n')

        return search_results

    def score(self, doc_bow, query_bow):
        total = 0.
        max_hits = max(doc_bow.values())
        for vocab_id, frequency in query_bow.items():
            term_hits = doc_bow[vocab_id] if vocab_id in doc_bow else 0
            doc_count = self.doc_freqs[vocab_id] if vocab_id in self.doc_freqs else 0
            tf = 0.5 + 0.5 * (term_hits / max_hits)
            idf = math.log(self.total_docs / (doc_count + 1))
            tf_idf = tf * idf
            total += frequency * tf_idf
        return total

    def create_bow(self, text, remove_stopwords=False):
        assert text is not None
        tokens = [x for x in self.nlp(text) if not (x.is_space or x.is_punct)]
        if remove_stopwords:
            tokens = [x for x in tokens if not x.is_stop]
        tokens = [x.lower_ for x in tokens]
        unigram_ids = [self.vocab[x] if x in self.vocab else self.unknown_id for x in tokens]
        bigram_ids = [tuple(unigram_ids[i:i+2]) for i in range(len(unigram_ids)-1)]
        token_ids = unigram_ids + bigram_ids
        token_ids = [str(x) for x in token_ids]
        bow = defaultdict(int)
        for token_id in token_ids:
            bow[token_id] += 1
        return bow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--docs_dir', required=True)
    parser.add_argument('-bd', '--bow_dir', required=True)
    parser.add_argument('-ip', '--index_path', required=True)
    parser.add_argument('-vp', '--vocab_path', required=True)
    parser.add_argument('-df', '--doc_freqs_path', required=True)
    parser.add_argument('-ii', '--inverted_index_path', required=True)
    parser.add_argument('-imb', '--in_memory_bow', action='store_true')
    parser.add_argument('-q', '--query')
    parser.add_argument('-iil', '--inverted_index_limit', type=int, default=1000)
    parser.add_argument('-rl', '--result_limit', type=int, default=10)
    parser.add_argument('-nt', '--num_threads', type=int, default=32)
    parser.add_argument('-as', '--add_snippet', action='store_true')
    parser.add_argument('-rs', '--remove_stopwords', action='store_true')
    parser.add_argument('-hp', '--hotpot_file')
    parser.add_argument('-of', '--output_path')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    nlp = spacy.load('en')
    print('Loading index from {}'.format(args.index_path))
    with open(args.index_path) as f:
        index = json.load(f)
    doc_store = DocStore(dir=args.docs_dir, index=index)

    if args.in_memory_bow:
        print('Loading in memory bow store from {}'.format(args.bow_dir))
        start = time.time()
        bow_store = InMemoryDocStore.load(input_dir=args.bow_dir, num_threads=args.num_threads)
        elapsed = time.time() - start
        print('Loading bow store complete in {}s'.format(elapsed))
    else:
        bow_store = DocStore(dir=args.bow_dir, index=index)

    print('Loading vocab from {}'.format(args.vocab_path))
    with open(args.vocab_path) as f:
        vocab = json.load(f)
    print('Loading document frequencies from {}'.format(args.doc_freqs_path))
    with open(args.doc_freqs_path) as f:
        doc_freqs = json.load(f)
    print('Loading inverted index from {}'.format(args.inverted_index_path))
    with open(args.inverted_index_path) as f:
        token_id_to_doc_ids = json.load(f)
        inverted_index = InvertedIndex(token_id_to_doc_ids=token_id_to_doc_ids)

    print('Finished loading data, creating model')
    model = BigramModel(nlp=nlp, doc_store=doc_store, bow_store=bow_store,
                        vocab=vocab, doc_freqs=doc_freqs, inverted_index=inverted_index)

    if args.hotpot_file is not None:
        assert args.output_path is not None
        print('Executing queries for hotpot file {}'.format(args.hotpot_file))
        with open(args.hotpot_file) as f:
            examples = json.load(f)
        results = []
        for example in tqdm(examples):
            id = example['_id']
            question = example['question']
            search_results = model.search(query=question,
                                          inverted_index_limit=args.inverted_index_limit,
                                          result_limit=args.result_limit,
                                          add_snippet=args.add_snippet,
                                          remove_stopwords=args.remove_stopwords,
                                          debug=args.debug)
            example_result = {'_id': id, 'question': question, 'results': search_results}
            results.append(example_result)
        print('Writing hotpot results to {}'.format(args.output_path))
        with open(args.output_path, 'w') as f:
            json.dump(results, f)
        print('Success!')
    elif args.query is not None:
        print('Executing query: {}'.format(args.query))
        start = time.time()
        search_results = model.search(query=args.query,
                                      inverted_index_limit=args.inverted_index_limit,
                                      result_limit=args.result_limit,
                                      add_snippet=args.add_snippet,
                                      remove_stopwords=args.remove_stopwords,
                                      debug=args.debug)
        elapsed = time.time() - start
        print('Finished search in {}s'.format(elapsed))

        if args.add_snippet:
            row_format = '{:<10}{:<10}{:<30}{:<50}{:<50}'
            print('Results:')
            print(row_format.format('score', 'id', 'title', 'url', 'text'))
            for search_result in search_results:
                doc = search_result['doc']
                score = search_result['score']
                score = '{:0.2f}'.format(score)
                text = ''.join(doc['text'][1])[:50]
                print(row_format.format(score, doc['id'], doc['title'], doc['url'], text))
        else:
            row_format = '{:<10}{:<10}'
            print('Results:')
            print(row_format.format('score', 'id'))
            for search_result in search_results:
                score = search_result['score']
                score = '{:0.2f}'.format(score)
                id = search_result['id']
                print(row_format.format(score, id))


if __name__ == '__main__':
    main()
