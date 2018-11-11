import argparse
import bz2
import json
import multiprocessing as mp
import os
import pickle
import spacy
import time
from tqdm import tqdm


class Tokenizer:
    def __init__(self, token_to_id, nlp=None, unknown_id=None):
        assert isinstance(token_to_id, dict)
        self.token_to_id = token_to_id
        if nlp is None:
            nlp = spacy.load('en')
        self.nlp = nlp
        if unknown_id is None:
            unknown_id = len(token_to_id)
        self.unknown_id = unknown_id

    def token_ids(self, text):
        assert isinstance(text, str)
        tokens = self.tokenize(text, self.nlp)
        return [self.token_to_id[x] if x in self.token_to_id else self.unknown_id for x in tokens]

    @classmethod
    def tokenize(cls, text, nlp):
        assert isinstance(text, str)
        assert nlp is not None
        unigrams = [x.lower_ for x in nlp(text) if not x.is_space]
        # bigrams = [tuple(unigrams[i:i+2]) for i in range(len(unigrams)-1)]
        return unigrams


class TokenizeRunner:
    def __init__(self, input_path, output_path):
        assert isinstance(input_path, str)
        assert isinstance(output_path, str)
        self.input_path = input_path
        self.output_path = output_path

    def __str__(self):
        return 'Running TokenizeRunner(input_path={}, output_path={})'.format(self.input_path, self.output_path)

    def run(self):
        print('Running {}'.format(self))
        start = time.time()
        nlp = spacy.load('en')
        docs = self.read_docs(self.input_path)
        for doc in docs:
            title = doc['title']
            body = doc['body']
            title_tokens = Tokenizer.tokenize(text=title, nlp=nlp)
            body_tokens = Tokenizer.tokenize(text=body, nlp=nlp)
            doc['title'] = title_tokens
            doc['body'] = body_tokens
        self.write_docs(path=self.output_path, docs=docs)
        elapsed = time.time() - start
        print('* Tokenized {} docs from {} in {}s'.format(len(docs), self.input_path, elapsed))

    @staticmethod
    def read_docs(path):
        assert isinstance(path, str)
        assert path.endswith('.bz2')
        with bz2.open(path) as f:
            return [json.loads(l) for l in f.readlines()]

    @staticmethod
    def write_docs(path, docs):
        assert isinstance(path, str)
        assert path.endswith('.bz2')
        assert isinstance(docs, list)
        docs_json = [json.dumps(doc) for doc in docs]
        docs_jsonlines = ''.join([x + '\n' for x in docs_json])
        docs_bytes = docs_jsonlines.encode('utf-8')
        with bz2.open(path, 'w') as f:
            f.write(docs_bytes)

    @classmethod
    def _run(cls, instance):
        assert isinstance(instance, cls)
        return instance.run()

    @classmethod
    def tokenize_docs(cls, input_dir, output_dir, num_threads):
        assert isinstance(input_dir, str)
        assert isinstance(output_dir, str)
        assert num_threads > 0

        start = time.time()
        print('Tokenizing docs in {}'.format(input_dir))
        input_paths = []
        for root, dir, files in os.walk(input_dir):
            for file in files:
                input_paths.append(os.path.join(root, file))
        assert all([x.startswith(input_dir) for x in input_paths])
        output_paths = [x.replace(input_dir, output_dir, 1) for x in input_paths]
        assert all([x.startswith(output_dir) and x.endswith('.bz2') for x in output_paths])

        print('Creating output directories')
        output_dirs = set(os.path.dirname(x) for x in output_paths)
        for dir in output_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        print('Creating {} tokenize runners'.format(len(input_paths)))
        runners = []
        for i in range(len(input_paths)):
            input_path = input_paths[i]
            output_path = output_paths[i]
            runner = TokenizeRunner(input_path=input_path, output_path=output_path)
            runners.append(runner)

        print('Executing runners')
        with mp.Pool(num_threads) as pool:
            pool.map(cls._run, runners)

        elapsed = time.time() - start
        print('Finished in {}s'.format(elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-t', '--num_threads', type=int, default=32)
    args = parser.parse_args()
    TokenizeRunner.tokenize_docs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_threads=args.num_threads)


if __name__ == '__main__':
    main()
