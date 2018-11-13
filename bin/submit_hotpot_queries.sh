#! /bin/bash

set -ex

function usage() {
  echo "Usage: submit_tokenize.sh <subdir>"
  exit 1
}

if [ -z "$1" ]; then
  usage
fi

path="$1"
tmpPath="output/$path"
hotpotPath="hotpot_chunked/$path"
outputPath="gs://hotpot-ir/data/output/tfidf_bigram/$path"

cd /dev/shm
if [ ! -d "output" ]; then
  mkdir output
fi
python3 hotpot_ir/bigram_model.py \
  --docs_dir na \
  --bow_dir wiki-tokenized-bow \
  --index_path wiki-index.json \
  --vocab_path vocab.json \
  --doc_freqs_path doc-freq.json \
  --inverted_index_path wiki-inverted-index.json \
  --in_memory_bow \
  --inverted_index_limit 5000 \
  --result_limit 5000 \
  --num_threads 32 \
  --hotpot_file $hotpotPath \
  --output_path $tmpPath
/snap/bin/gsutil cp $tmpPath $outputPath
