import argparse
import bz2
from collections import defaultdict
import json
import multiprocessing as mp
import pickle
import os
import time
from tqdm import tqdm


class StatsCalculator:
    def __init__(self, input_path):
        assert isinstance(input_path, str)
        self.input_path = input_path

    def __str__(self):
        return 'StatsCalculator(input_path={})'.format(self.input_path)

    def run(self):
        print('Running {}'.format(self))
        start = time.time()
        docs = self.read_docs()
        doc_freq = self.calculate_document_frequency(docs)
        elapsed = time.time() - start
        print('* Finished {} in {}s'.format(self, elapsed))
        return doc_freq

    def calculate_document_frequency(self, docs):
        assert isinstance(docs, list)
        doc_freq = defaultdict(int)
        for doc in docs:
            bow = doc['bow']
            assert isinstance(bow, dict)
            for token, count in bow.items():
                if count > 0:
                    doc_freq[token] += 1
        doc_freq['TOTAL'] = len(docs)
        return doc_freq

    def read_docs(self):
        with bz2.open(self.input_path) as f:
            return [json.loads(l) for l in f.readlines()]

    @classmethod
    def _run(cls, instance):
        assert isinstance(instance, cls)
        return instance.run()

    @classmethod
    def merge(cls, *doc_freqs):
        assert all([isinstance(x, dict) for  x in doc_freqs])
        merged = defaultdict(int)
        for doc_freq in tqdm(doc_freqs):
            for token, doc_count in doc_freq.items():
                merged[token] += doc_count
        return merged


    @classmethod
    def calculate(cls, input_dir, num_threads, num_chunks=10):
        assert isinstance(input_dir, str)
        input_paths = []
        for root, dir, files in os.walk(input_dir):
            for file in files:
                input_paths.append(os.path.join(root, file))
        input_paths = sorted(input_paths)

        print('Calculating stats for {} input paths...'.format(len(input_paths)))
        calculators = [cls(x) for x in input_paths]
        chunk_size = max(1, int(len(calculators) / num_chunks))
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:
                chunk = calculators[start_idx:]
            else:
                end_idx = start_idx + chunk_size
                chunk = calculators[start_idx:end_idx]
            chunks.append(chunk)
        assert sum([len(x) for x in chunks]) == len(input_paths)

        print('Using {} chunks of size {}'.format(num_chunks, chunk_size))
        with mp.Pool(num_threads) as pool:
            merged = {}
            for i, chunk in enumerate(chunks):
                print('** Processing chunk {}'.format(i))
                results = pool.map(cls._run, chunk)
                print('** Aggregating stats')
                merged = cls.merge(merged, *results)

        return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-t', '--num_threads', type=int, default=8)
    parser.add_argument('-c', '--num_chunks', type=int, default=10)
    args = parser.parse_args()

    doc_freq = StatsCalculator.calculate(input_dir=args.input_dir,
                                         num_threads=args.num_threads,
                                         num_chunks=args.num_chunks)
    with open(args.output_path, 'w') as f:
        json.dump(doc_freq, f)
    print('Wrote results to {}'.format(args.output_path))


if __name__ == '__main__':
    main()
