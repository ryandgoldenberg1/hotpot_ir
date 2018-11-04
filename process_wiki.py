import argparse
import bs4
import bz2
import json
import os
import time
from tqdm import tqdm


def walk_dir(dir):
    for root, dir, files in os.walk(dir):
        for file in files:
            yield os.path.join(root, file)


def read_docs(wiki_dir):
    total_paths = 0
    for path in walk_dir(wiki_dir):
        total_paths += 1

    for path in tqdm(walk_dir(wiki_dir), total=total_paths, unit='files'):
        with bz2.open(path) as f:
            data = f.read()
            lines = data.split(b'\n')
            for line in lines:
                line = line.strip()
                if (len(line) == 0):
                    continue
                doc = json.loads(line)
                yield doc


def process(doc):
    id = doc['id']
    url = doc['url']
    title = doc['title']
    paragraphs = doc['text']
    assert id is not None and len(id) > 0
    assert url is not None and len(url) > 0
    assert title is not None and len(title) > 0
    assert paragraphs is not None and len(paragraphs) > 0
    assert ''.join(paragraphs[0]) == title
    paragraphs = paragraphs[1:]

    sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
    text = ''.join(sentences)
    soup = bs4.BeautifulSoup(text, features='html5lib')
    body = title + '. ' + soup.text
    links = [{'href': x.get('href'), 'text': x.text}  for x in soup.find_all('a')]

    result = {
        'id': doc['id'],
        'url': doc['url'],
        'title': title,
        'body': body,
        'links': links,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wiki_dir', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    args = parser.parse_args()

    processed_docs = 0
    for doc in read_docs(args.wiki_dir):
        result = process(doc)
        result_bytes = json.dumps(result).encode('utf-8')
        output_path = os.path.join(args.output_dir, result['id'] + '.bz2')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with bz2.open(output_path, 'w') as f:
            f.write(result_bytes)
        processed_docs += 1
    print('Processed {} docs'.format(processed_docs))
    print('Wrote results to: {}'.format(args.output_dir))



if __name__ == '__main__':
    main()
