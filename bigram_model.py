import argparse
from collections import defaultdict
import doc_store as ds
import json
import math
import process_wiki as pw
import spacy


class BigramModel:
    def __init__(self, nlp, doc_store):
        assert nlp is not None
        assert doc_store is not None
        self.nlp = nlp
        self.doc_store = doc_store

    def search(self, query, limit):
        assert query is not None
        assert limit > 0
        docs = [x for x in self.doc_store.all()]
        doc_scores = [(x, self.score(x, query)) for x in docs]
        doc_scores = sorted(doc_scores, key=lambda x: -x[1])
        return doc_scores[:limit]

    def score(self, doc, query):
        assert doc is not None and doc['body'] is not None
        assert query is not None
        assert self.document_counts_by_vocab_id is not None
        assert self.total_documents is not None
        doc_bow = self.vocab_id_frequencies(doc['body'])
        query_bow = self.vocab_id_frequencies(query)
        total = 0.
        for vocab_id, frequency in query_bow.items():
            term_hits = doc_bow[vocab_id]
            assert term_hits >= 0
            doc_count = self.document_counts_by_vocab_id[vocab_id]
            idf = math.log(self.total_documents / (doc_count + 1))
            tf_idf = term_hits * idf
            total += frequency * tf_idf
        return total

    def vocab_id_frequencies(self, text):
        assert text is not None
        assert self.vocab is not None
        tokens = self.tokenize(text)
        vocab_ids = [self.vocab[t] if t in self.vocab else -1 for t in tokens]
        frequencies = defaultdict(int)
        for vocab_id in vocab_ids:
            frequencies[vocab_id] += 1
        return frequencies

    def tokenize(self, text):
        assert text is not None
        tokens = [x.lower_ for x in self.nlp(text)]
        bigram_tokens = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        return bigram_tokens

    def preprocess(self):
        token_counts = defaultdict(int)
        document_counts = defaultdict(int)
        total_documents = 0
        for doc in self.doc_store.all():
            total_documents += 1
            tokens = self.tokenize(doc['body'])
            for token in tokens:
                token_counts[token] += 1
            for token in set(tokens):
                document_counts[token] += 1
        tokens = list(token_counts.keys())
        tokens = sorted(tokens, key=lambda x: -token_counts[x])
        vocab = {tokens[i]: i for i in range(len(tokens))}
        document_counts_by_vocab_id = defaultdict(int)
        for token, count in document_counts.items():
            document_counts_by_vocab_id[vocab[token]] = count
        self.vocab = vocab
        self.document_counts_by_vocab_id = document_counts_by_vocab_id
        self.total_documents = total_documents

    @classmethod
    def build(cls, nlp, doc_store):
        model = BigramModel(nlp=nlp, doc_store=doc_store)
        model.preprocess()
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-q', '--query', required=True)
    parser.add_argument('-l', '--limit', type=int, default=10)
    args = parser.parse_args()

    nlp = spacy.load('en')
    doc_store = ds.DocStore(args.input_dir)
    print('Building model...')
    model = BigramModel.build(nlp=nlp, doc_store=doc_store)

    print('Searching for query: {}...'.format(args.query))
    results = model.search(query=args.query, limit=args.limit)
    print('Results:')
    for doc, score in results:
        print('\t{}\t{:.2f}'.format(doc['title'], score))

if __name__ == '__main__':
    main()
