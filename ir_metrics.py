"""
Calculates json
"""

import argparse
import json
import numpy as np

# JSONS = ['hotpot/hotpot_dev_distractor_v1.json',
#          'hotpot/hotpot_dev_fullwiki_v1.json']
JSONS = ['hotpot/hotpot_dev_fullwiki_v1.json']
parser = argparse.ArgumentParser(description='get IR metrics')
parser.add_argument('--json', help='JSON file(s)', nargs='*', default=JSONS)
parser.add_argument('--output', help='output file name', default='output.txt')

def get_idx(hit, article_titles):
    try:
        return article_titles.index(hit)
    except:
        return 5001
        # return len(article_titles) + 1

def get_ir_metrics(dataset, max_K=10):
    """ Get IR metrics for the given dataset. Note: mean rank is not an accurate metric unless 5000
        documents are passed in, as they did for HotPotQA.
    Arguments:
        dataset (dict) -- dataset loaded from JSON
        max_K (int)-- max value of K
    Returns:
        float, int, list(int) -- MAP, mean rank, hits at K
    """
    total_num_hits = 0
    total_num_gold = len(dataset) * 2
    hits_at = [[] for x in range(10)]
    prec_at = [[] for x in range(len(dataset))]
    avg_ranks = []
    for i, question_item in enumerate(dataset):
        context = question_item['context']
        article_titles = [x[0] for x in context]
        supporting_facts = question_item['supporting_facts']
        gold_titles = set([x[0] for x in supporting_facts])

        # if len(gold_titles) != 2: # should only have 2 gold articles
        #     import pdb; pdb.set_trace()

        num_relevant = 0
        max_range = min(len(context), max_K)
        for K in range(1, max_range + 1): # get precision@K
            article_titlesK = article_titles[:K]
            is_relevant = article_titles[K - 1] in gold_titles
            hits = set(article_titlesK).intersection(gold_titles)
            num_hits = len(hits)
            hits_at[K - 1].append(num_hits / min(K, 2))
            if not is_relevant:
                continue
            prec_at[i].append(num_hits / K)
        hits_idx = [get_idx(x, article_titles) for x in gold_titles]
        avg_idx = np.mean(hits_idx)
        avg_ranks.append(avg_idx)

    prec_at = [x for x in prec_at if x]
    avg_prec = [np.mean(x) for x in prec_at]
    mean_avg_prec = np.mean(avg_prec)
    mean_rank = np.mean(avg_ranks)
    mean_hits_at = [np.mean(x) for x in hits_at]
    print(mean_avg_prec)
    print(mean_rank)
    print(mean_hits_at)

    return(mean_avg_prec, mean_rank, hits_at)


def main():
    args = parser.parse_args()
    for json_name in args.json:
        print('loading dataset from {}'.format(json_name))
        with open(json_name) as f:
            dataset = json.load(f)
        metrics = get_ir_metrics(dataset)

if __name__ == '__main__':
    main()