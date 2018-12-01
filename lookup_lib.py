"""
Contains class that allow for efficient lookup of documents given id
"""

import argparse
import glob
import hashlib
try:
    import ujson as json
except:
    print('could not find ujson, using json')
    import json
from nltk import tokenize
import os
import pandas as pd

parser = argparse.ArgumentParser(description='lookup documents given id')

parser.add_argument('--wiki', default='docs_parsed', help='path to wiki dump location')
parser.add_argument('--dict', default='/home/bl2557/hotpot_ir/lookup_d.json', help='pre-extracted lookup dictionary')

class DocumentLookup(object):
    def __init__(self, wiki_dump=None, lookup_dict=None):
        self.wiki_dump = wiki_dump
        if os.path.exists(lookup_dict):
            print('loading pre-extracted lookup dictionary from {}'.format(lookup_dict))
            with open(lookup_dict) as f:
                self.d = json.load(f)
        else:
            self.d = {}
            self.populate_dict()

        self.hash2id = {}
        self.gen_hash2id()

    def gen_hash2id(self):
        print('adding hash ids to lookup_d...')
        for id, data in self.d.items():
            hashed = hashlib.sha1(data[0].encode('utf-8')).hexdigest()
            self.hash2id[hashed] = id

    def populate_dict(self):
        files = glob.glob('{}/*.parquet'.format(self.wiki_dump))
        print('found {} files to populate dict with'.format(len(files)))
        for fname in files:
            print('processing {}...'.format(fname))
            df = pd.read_parquet(fname, columns=['id','title','text_parsed'])
            df['title_len'] = df.title.str.len()
            df['text_new'] = df.apply(lambda x: x['text_parsed'][x['title_len'] + 1:], 1) # remove title name
            df['sentences'] = df['text_parsed'].apply(tokenize.sent_tokenize)
            for tup in df.itertuples():
                self.d[tup.id] = (tup.title, tup.sentences)

    def get(self, id):
        return self.d.get(id)

    def get_hashed_id(self, hashed_id):
        id = self.hash2id[hashed_id]
        return self.d.get(id)

if __name__ == "__main__":
    args = parser.parse_args()
    lookup = DocumentLookup(args.wiki, args.dict)
