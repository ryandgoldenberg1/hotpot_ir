"""
Creates files that allow for article_id -> processed text mapping.
Run this, then run article_lib.py.
"""

import argparse
import bs4
import bz2
import pickle
import json
import multiprocessing as mp
import os
import time


class ArticleParser:
    def __init__(self, input_path, output_path):
        assert isinstance(input_path, str)
        assert isinstance(output_path, str)
        self.input_path = input_path
        self.output_path = output_path

    def __str__(self):
        return 'ArticleParser(input_path={}, output_path={})'.format(self.input_path, self.output_path)

    def run(self):
        lookup_dict = {}
        print('Running parser {}'.format(self))
        start = time.time()
        docs = self.read_docs(self.input_path)
        parsed_docs = [self.parse(doc) for doc in docs]
        elapsed = time.time() - start
        print('* Finished parsing {} docs in {}s for path {}'.format(len(parsed_docs), elapsed, self.input_path))
        for d in parsed_docs:
            lookup_dict[d['id']] = [d['title'], d['sentences']]
        self.write_docs(self.output_path, parsed_docs)

    @staticmethod
    def parse(doc):
        id = doc['id']
        url = doc['url']
        title = doc['title']
        paragraphs = doc['text']
        assert id is not None and len(id) > 0
        assert url is not None and len(url) > 0
        assert title is not None and len(title) > 0
        assert paragraphs is not None and len(paragraphs) > 0
        if not ''.join(paragraphs[0]) == title:
            print('[WARN] title does not match first paragraph for document: {}'.format(doc))
        paragraphs = paragraphs[1:]
        sentences = []
        num_chars = 0
        for paragraph in paragraphs:
            for sentence in paragraph:
                text = bs4.BeautifulSoup(sentence, features='html5lib').text
                num_chars += len(text)
                sentences.append(text)
            if num_chars >= 500:
                break
        return {'id': id, 'title': title, 'sentences': sentences}

    @staticmethod
    def read_docs(path):
        assert isinstance(path, str)
        assert path.endswith('.bz2')
        with bz2.open(path) as f:
            return [json.loads(l.decode('utf-8')) for l in  f.readlines()]

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
    
    @staticmethod
    def reconstruct_dictionary(parsers):
        d = {}
        for parser in parsers:
            docs = parser.read_docs(parser.output_path)
            for doc in docs:
                d[doc['id']] = [doc['title'], doc['sentences']]
        return d
            
    @classmethod
    def parse_all(cls, input_dir, output_dir, lookup_dict_path=None, num_threads=32):
        assert isinstance(input_dir, str)
        assert isinstance(output_dir, str)
        assert num_threads > 0
        input_paths = []
        for root, dir, files in os.walk(input_dir):
            for file in files:
                input_paths.append(os.path.join(root, file))
        print('Found {} input paths for dir {}'.format(len(input_paths), input_dir))
        assert all([x.startswith(input_dir) for x in input_paths])
        output_paths = [x.replace(input_dir, output_dir, 1) for x in input_paths]
        assert all([x.startswith(output_dir) for x in output_paths])

        print('Creating output directories...')
        output_dirs = set(os.path.dirname(x) for x in output_paths)
        for dir in output_dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        print('Creating parsers...')
        parsers = []
        for i in range(len(input_paths)): 
            input_path = input_paths[i]
            output_path = output_paths[i]
            parser = cls(input_path=input_path, output_path=output_path)
            parsers.append(parser)
        # import pdb; pdb.set_trace()
        print('Executing parsers...')
        if num_threads > 1:
            with mp.Pool(num_threads) as pool:
                pool.map(cls._run, parsers)
        else:
            for parser in parsers:
                parser.run()
                
        # lookup_dict = cls.reconstruct_dictionary(parsers)
        # print('saving lookup_dict to {}'.format(lookup_dict_path))
        # with open(lookup_dict_path, 'wb') as f:
            # pickle.dump(lookup_dict, f)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='/home/rg3155/wiki/enwiki-20171001-pages-meta-current-withlinks-processed/')
    parser.add_argument('-o', '--output_dir', default='/home/rg3155/wiki/articles/')
    parser.add_argument('-d', '--lookup_dict', default='lookup.pkl')
    parser.add_argument('-t', '--num_threads', type=int, default=32)
    args = parser.parse_args()
    ArticleParser.parse_all(args.input_dir, args.output_dir, args.lookup_dict, args.num_threads)
    


if __name__ == '__main__':
    main()
