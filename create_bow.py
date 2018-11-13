import argparse
import os
import time
import multiprocessing as mp
import bz2
import json
from collections import defaultdict


class BowConverter:
    def __init__(self, vocab, input_path, output_path):
        assert isinstance(vocab, dict)
        assert isinstance(input_path, str)
        assert isinstance(output_path, str)
        self.vocab = vocab
        self.input_path = input_path
        self.output_path = output_path

    def __str__(self):
        return 'BowConverter(input_path={}, output_path={})'.format(self.input_path, self.output_path)

    def run(self):
        print('Starting converter {}'.format(self))
        start = time.time()
        docs = self.read_docs()
        docs = [self.convert(doc) for doc in docs]
        self.write_docs(docs)
        elapsed = time.time() - start
        print('* Finished {} in {}s'.format(self, elapsed))

    def convert(self, doc):
        assert isinstance(doc, dict)
        bow = defaultdict(int)
        body = doc['body']
        unigrams = [self.vocab[x] for x in body]
        bigrams = [str(tuple(unigrams[i:i+2])) for i in range(len(unigrams) - 1)]
        ngrams = unigrams + bigrams
        for ngram in ngrams:
            bow[ngram] += 1
        return {'id': doc['id'], 'bow': bow}

    def read_docs(self):
        with bz2.open(self.input_path) as f:
            return [json.loads(l) for l in f.readlines()]

    def write_docs(self, docs):
        docs_json = [json.dumps(doc) for doc in docs]
        docs_jsonlines = ''.join([x + '\n' for x in docs_json])
        docs_bytes = docs_jsonlines.encode('utf-8')
        with bz2.open(self.output_path, 'w') as f:
            f.write(docs_bytes)

    @classmethod
    def _run(cls, instance):
        assert isinstance(instance, cls)
        return instance.run()

    @classmethod
    def convert_all(cls, vocab, input_dir, output_dir, num_threads, chunk_size):
        assert isinstance(vocab, dict)
        assert isinstance(input_dir, str)
        assert isinstance(output_dir, str)
        assert num_threads > 0
        assert chunk_size > 0

        input_paths = []
        for root, dir, files in os.walk(input_dir):
            for file in files:
                input_paths.append(os.path.join(root, file))
        input_paths = sorted(input_paths)
        output_paths = [x.replace(input_dir, output_dir, 1) for x in input_paths]
        assert all([x.startswith(output_dir) for x in output_paths])
        output_dirs = set(os.path.dirname(x) for x in output_paths)
        for dir in output_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        converters = []
        for i in range(len(input_paths)):
            input_path = input_paths[i]
            output_path = output_paths[i]
            converter = cls(vocab=vocab, input_path=input_path, output_path=output_path)
            converters.append(converter)

        num_chunks = max(1, int(len(converters) / chunk_size))
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:
                chunk = converters[start_idx:]
            else:
                end_idx = start_idx + chunk_size
                chunk = converters[start_idx:end_idx]
            chunks.append(chunk)

        with mp.Pool(num_threads) as pool:
            for i in range(len(chunks)):
                chunk = chunks[i]
                chunk_start = time.time()
                chunk_paths = [x.input_path for x in chunk]
                print('chunk_paths={}'.format(chunk_paths))
                pool.map(cls._run, chunk)
                chunk_elapsed = time.time() - chunk_start
                print('## Finished Chunk {} in {}s'.format(i, chunk_elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-v', '--vocab_path', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-t', '--num_threads', type=int, default=10)
    parser.add_argument('-c', '--chunk_size', type=int, default=100)
    args = parser.parse_args()

    with open(args.vocab_path) as f:
        vocab = json.load(f)
    print('Loaded vocab of size {} from {}'.format(len(vocab), args.vocab_path))
    BowConverter.convert_all(
        vocab=vocab,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_threads=args.num_threads,
        chunk_size=args.chunk_size)
    print('Wrote bows to {}'.format(args.output_dir))



if __name__ == '__main__':
    main()
