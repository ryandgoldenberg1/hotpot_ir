"""
Takes raw IR results JSONs and formats it to be the same as the provided JSON sets.
"""

import argparse
import glob
import json
import os
import pandas as pd

import lookup_lib

RESULTS = [
    'results/distractor.json',
    'results/full_wiki.json',
    'results/train.json',
]

IN_FILES = [
    'hotpot/hotpot_dev_distractor_v1.json',
    'hotpot/hotpot_dev_fullwiki_v1.json',
    'hotpot/hotpot_train_v1.json',
]

OUT_FILES = [
    'results/distractor_formatted.json',
    'results/full_wiki_formatted.json',
    'results/train_formatted.json',
]

parser = argparse.ArgumentParser(description='format IR results for evaluation')

parser.add_argument('--results', default=RESULTS, nargs='*', help='result file(s)')
parser.add_argument('--input', default=IN_FILES, nargs='*', help='provided input files')
parser.add_argument('--output', default=OUT_FILES, nargs='*', help='output files')
parser.add_argument('--dict', default='lookup_d.json', help='pre-extracted lookup dictionary')

def main(args):
    docs = lookup_lib.DocumentLookup(lookup_dict=args.dict)
    for res_path, in_path, out_path in zip(args.results, args.input, args.output):
        print('processing {}...'.format(res_path))
        with open(res_path) as res_f, open(in_path) as in_f, open(out_path, 'w') as out_f:
            res = json.load(res_f)
            inp = json.load(in_f)
            out_d = {x['_id']: x for x in inp}
            for qa_item in out_d.values():
                qa_item['context'] = []
            for search in res:
                query_id = search['query_id']
                doc_id = search['doc_id']
                # doc_info = list(docs.get(doc_id))
                doc_info = [docs.get(doc_id)[0], [], search['score']]
                # import pdb; pdb.set_trace()
                out_d[query_id]['context'].append(doc_info)
            print('saving to {}...'.format(out_path))
            out_vals = list(out_d.values())
            json.dump(out_vals, out_f)
            # import pdb; pdb.set_trace()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
