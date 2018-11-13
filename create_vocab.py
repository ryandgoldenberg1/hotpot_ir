import argparse
import bz2
from collections import defaultdict
import json
import multiprocessing as mp
import pickle
import os
import time


class VocabGenerator:
    def __init__(self, path):
        assert isinstance(path, str)
        self.path = path

    def run(self):
        print('Counting tokens for {}'.format(self.path))
        start = time.time()
        docs = self.read_docs(self.path)
        token_counts = defaultdict(int)
        for doc in docs:
            body = doc['body']
            assert isinstance(body, list)
            for token in body:
                token_counts[token] += 1
        elapsed = time.time() - start
        print('* Counted {} tokens for {} docs in {}s'.format(len(token_counts), len(docs), elapsed))
        return token_counts

    @staticmethod
    def read_docs(path):
        assert isinstance(path, str)
        assert path.endswith('.bz2')
        with bz2.open(path) as f:
            return [json.loads(l) for l in f.readlines()]

    @classmethod
    def merge_counts(cls, *token_counts):
        for t in token_counts:
            assert isinstance(t, dict)
        merged_token_counts = defaultdict(int)
        for t in token_counts:
            for token, count in t.items():
                merged_token_counts[token] += count
        return merged_token_counts

    @classmethod
    def _run(cls, instance):
        assert isinstance(instance, cls)
        return instance.run()

    @classmethod
    def create_vocab(cls, input_dir, num_threads):
        assert isinstance(input_dir, str)
        assert num_threads > 0

        print('Creating vocabulary for {}'.format(input_dir))
        start = time.time()

        for root, dir, files in os.walk(input_dir):
            if len(files) > 0:
                assert os.path.dirname(root) == input_dir
        subdirs = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
        assert all([os.path.isdir(x) for x in subdirs])

        print('Counting tokens...')
        token_counts = defaultdict(int)
        with mp.Pool(num_threads) as pool:
            for subdir in subdirs:
                paths = [os.path.join(subdir, x) for x in os.listdir(subdir)]
                assert all([os.path.isfile(x) for x in paths])
                generators = [cls(x) for x in paths]
                sub_token_counts = pool.map(cls._run, generators)
                token_counts = cls.merge_counts(token_counts, *sub_token_counts)
                print('###\nFinished {}\n###'.format(subdir))

        print('Creating vocabulary from token counts...')
        tokens = sorted(list(token_counts.keys()), key=lambda x: -token_counts[x])
        vocab = {tokens[i]: i for i in range(len(tokens))}

        elapsed = time.time() - start
        print('Finished in {}s'.format(elapsed))
        return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-v', '--vocab_path', required=True)
    parser.add_argument('-t', '--num_threads', type=int, default=32)
    args = parser.parse_args()

    vocab = VocabGenerator.create_vocab(input_dir=args.input_dir, num_threads=args.num_threads)
    with open(args.vocab_path, 'w') as f:
        json.dump(vocab, f)
    print('Wrote vocab to {}'.format(args.vocab_path))


if __name__ == '__main__':
    main()
