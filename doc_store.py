import argparse
import bz2
import json
import multiprocessing as mp
import os
import time
from tqdm import tqdm


def _read(path):
    with bz2.open(path) as f:
        return [json.loads(l) for l in f.readlines()]


class Indexer:
    def __init__(self, dir, path):
        assert isinstance(dir, str)
        assert isinstance(path, str)
        self.dir = dir
        self.path = path

    def __str__(self):
        return 'Indexer(dir={}, path={})'.format(self.dir, self.path)

    def run(self):
        start = time.time()
        full_path = os.path.join(self.dir, self.path)
        docs = _read(full_path)
        index = {doc['id']: self.path for doc in docs}
        elapsed = time.time() - start
        print('* Finished {} in {}s'.format(self, elapsed))
        return index

    @classmethod
    def merge(cls, *indices):
        assert all([isinstance(x, dict) for x in indices])
        merged = {}
        for index in tqdm(indices):
            for id, path in index.items():
                assert id not in merged, 'Duplicate id {}'.format(id)
                merged[id] = path
        return merged

    @classmethod
    def _run(cls, instance):
        assert isinstance(instance, cls)
        return instance.run()

    @classmethod
    def index(cls, input_dir, num_threads):
        assert isinstance(input_dir, str)
        indexers = []
        for root, dir, files in os.walk(input_dir):
            if len(files) > 0:
                assert os.path.dirname(root) == input_dir, 'File must be at depth two: {}'.format(root)
            dir = os.path.basename(root)
            assert len(dir) > 0
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(path=full_path, start=input_dir)
                indexer = cls(dir=input_dir, path=rel_path)
                indexers.append(indexer)
        print('Creating indices for {} paths...'.format(len(indexers)))
        with mp.Pool(num_threads) as pool:
            indices = pool.map(cls._run, indexers)
        print('Merging indices...')
        index = cls.merge(*indices)
        return index


class DocStore:
    def __init__(self, dir, index):
        assert dir is not None
        assert os.path.isdir(dir)
        assert isinstance(index, dict)
        self.dir = dir
        self.index = index

    def all(self):
        for root, dir, files in os.walk(self.dir):
            for file in files:
                path = os.path.join(root, file)
                yield self._read(path)

    def get(self, ids):
        assert ids is not None
        if isinstance(ids, str) or isinstance(ids, int):
            return self._get(ids)
        elif isinstance(ids, list):
            return self._get_all(ids)
            # return [self._get(id) for id in ids]
        else:
            raise ValueError('Invalid ids: {}'.format(ids))

    def _get_all(self, ids):
        paths = list(set(os.path.join(self.dir, self.index[id]) for id in ids))
        start = time.time()
        with mp.Pool(32) as pool:
            docs_list = pool.map(_read, paths)
        elapsed = time.time() - start
        print('Fetched {} docs in {}s'.format(len(ids), elapsed))
        docs_by_id = {}
        target_ids = set(ids)
        for docs in docs_list:
            for doc in docs:
                if doc['id'] in ids:
                    docs_by_id[doc['id']] = doc
        assert len(docs_by_id) == len(ids)
        results = [docs_by_id[id] for id in ids]
        return results

    def _get(self, id):
        assert isinstance(id, str) or isinstance(id, int)
        id = str(id)
        if id not in self.index:
            raise ValueError('id {} not found'.format(id))
        path = os.path.join(self.dir, self.index[id])
        docs = _read(path)
        for doc in docs:
            if doc['id'] == id:
                return doc
        raise ValueError('id {} not found in path {}'.format(id, path))


class InMemoryDocStore:
    def __init__(self, docs):
        assert isinstance(docs, dict)
        self.docs = docs

    def get(self, ids):
        if isinstance(ids, str) or isinstance(ids, int):
            return self.docs[str(ids)]
        elif isinstance(ids, list):
            return [self.docs[str(id)] for id in ids]
        else:
            raise ValueError('Invalid ids: {}'.format(ids))

    @classmethod
    def load(cls, input_dir, num_threads):
        assert isinstance(input_dir, str)
        input_paths = []
        for root, dir, files in os.walk(input_dir):
            for file in files:
                path = os.path.join(root, file)
                input_paths.append(path)
        print('Reading docs from {} paths'.format(len(input_paths)))
        with mp.Pool(num_threads) as pool:
            docs_list = pool.map(_read, input_paths)
        docs_by_id = {}
        print('Merging docs into single index')
        for docs in tqdm(docs_list):
            for doc in docs:
                docs_by_id[doc['id']] = doc
        return cls(docs=docs_by_id)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', required=True)
    parser.add_argument('-ip', '--index_path')
    parser.add_argument('-c', '--create_index', action='store_true')
    parser.add_argument('-nt', '--num_threads', type=int, default=32)
    parser.add_argument('-im', '--in_memory', action='store_true')
    parser.add_argument('-q', '--query')
    args = parser.parse_args()

    if args.in_memory:
        print('Loading InMemoryDocStore from {}'.format(args.input_dir))
        doc_store = InMemoryDocStore.load(input_dir=args.input_dir,
                                          num_threads=args.num_threads)
    else:
        assert args.index_path is not None
        if args.create_index:
            print('Creating index for {}'.format(args.input_dir))
            index = Indexer.index(input_dir=args.input_dir, num_threads=args.num_threads)
            print('Writing index to {}'.format(args.index_path))
            with open(args.index_path, 'w') as f:
                json.dump(index, f)
            print('Success!')
        else:
            print('Loading index from {}'.format(args.index_path))
            with open(args.index_path) as f:
                index = json.load(f)
        doc_store = DocStore(dir=args.input_dir, index=index)

    if args.query is not None:
        print('Executing query')
        start = time.time()
        doc = doc_store.get(args.query)
        elapsed = time.time() - start
        print('Retrieved result in {}s'.format(elapsed))
        snippet = {'id': doc['id'], 'title': doc['title'], 'text': doc['text'][:3]}
        print('Snippet: {}'.format(snippet))


if __name__ == '__main__':
    main()
