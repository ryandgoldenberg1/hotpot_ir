import argparse
import bz2
from collections import defaultdict
import json
import multiprocessing as mp
import os
import time
from tqdm import tqdm


class InvertedIndexBuilder:
    def __init__(self, path):
        assert isinstance(path, str)
        self.path = path

    def __str__(self):
        return 'InvertedIndexBuilder(path={})'.format(self.path)

    def run(self):
        start = time.time()
        docs = self.read()
        index = defaultdict(list)
        for doc in docs:
            id = doc['id']
            bow = doc['bow']
            assert isinstance(bow, dict)
            for token_id, _ in bow.items():
                index[token_id].append(id)
        elapsed = time.time() - start
        print('Finished {} in {}s'.format(self, elapsed))
        return index

    def read(self):
        with bz2.open(self.path) as f:
            return [json.loads(l) for l in f.readlines()]

    @classmethod
    def _run(cls, instance):
        assert isinstance(instance, cls)
        return instance.run()

    @classmethod
    def merge(cls, *indices):
        assert all([isinstance(x, dict) for x in indices])
        merged = defaultdict(list)
        for index in tqdm(indices):
            for token_id, doc_ids in index.items():
                merged[token_id] += doc_ids
        return merged

    @classmethod
    def build(cls, bow_dir, num_threads, chunk_size):
        assert isinstance(bow_dir, str)
        assert num_threads > 0
        assert chunk_size > 0
        instances = []
        for root, dir, files in os.walk(bow_dir):
            for file in files:
                path = os.path.join(root, file)
                instance = cls(path=path)
                instances.append(instance)
        instances = sorted(instances, key=lambda x: x.path)
        chunks = []
        num_chunks = max(1, int(len(instances)/chunk_size))
        print('Using {} chunks with size {}'.format(num_chunks, chunk_size))
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:
                chunk = instances[start_idx:]
            else:
                end_idx = start_idx + chunk_size
                chunk = instances[start_idx:end_idx]
            chunks.append(chunk)
        print('Building inverted index for {} paths'.format(len(instances)))
        index = defaultdict(list)
        for i, chunk in enumerate(chunks):
            print('Starting chunk {}'.format(i))
            with mp.Pool(num_threads) as pool:
                indices = pool.map(cls._run, chunk)
            print('Aggregating indices...')
            index = cls.merge(index, *indices)
            indices = []
        return index


class InvertedIndex:
    def __init__(self, token_id_to_doc_ids):
        assert isinstance(token_id_to_doc_ids, dict)
        self.token_id_to_doc_ids = token_id_to_doc_ids

    def get(self, bow, limit):
        assert isinstance(bow, dict)
        assert limit > 0
        doc_id_to_score = defaultdict(int)
        for token_id, count in bow.items():
            assert count > 0
            if token_id not in self.token_id_to_doc_ids:
                continue
            doc_ids = self.token_id_to_doc_ids[token_id]
            for doc_id in doc_ids:
                doc_id_to_score[doc_id] += count
        doc_ids = sorted(doc_id_to_score.keys(), key=lambda x: -doc_id_to_score[x])
        return doc_ids[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--inverted_index_path', required=True)
    parser.add_argument('-bd', '--bow_dir')
    parser.add_argument('-nt', '--num_threads', type=int, default=32)
    parser.add_argument('-cs', '--chunk_size', type=int, default=100)
    parser.add_argument('-c', '--create', action='store_true')
    parser.add_argument('-q', '--query')
    parser.add_argument('-vp', '--vocab_path')
    parser.add_argument('-l', '--limit', type=int, default=10)
    args = parser.parse_args()

    if args.create:
        print('Creating inverted index')
        inverted_index = InvertedIndexBuilder.build(bow_dir=args.bow_dir, num_threads=args.num_threads, chunk_size=args.chunk_size)
        print('Writing inverted index to {}'.format(args.inverted_index_path))
        with open(args.inverted_index_path, 'w') as f:
            json.dump(inverted_index, f)
        print('Success!')
    else:
        print('Loading inverted index from {}'.format(args.inverted_index_path))
        with open(args.inverted_index_path) as f:
            inverted_index = json.load(f)
    inverted_index = InvertedIndex(inverted_index)

    if args.query is not None:
        assert args.vocab_path is not None
        print('Loading vocab from {}'.format(args.vocab_path))
        with open(args.vocab_path) as f:
            vocab = json.load(f)
        print('Executing query: {}'.format(args.query))
        tokens = [x.lower() for x in args.query.split()]
        token_ids = [vocab[x] for x in tokens]
        bow = defaultdict(int)
        for token_id in token_ids:
            bow[str(token_id)] += 1
        doc_ids = inverted_index.get(bow=bow, limit=args.limit)
        print('Found doc ids: {}'.format(doc_ids))
        print('Top match: https://en.wikipedia.org/wiki?curid={}'.format(doc_ids[0]))


if __name__ == '__main__':
    main()
