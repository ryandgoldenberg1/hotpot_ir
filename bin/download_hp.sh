#! /bin/bash

set -e

cd /dev/shm
/snap/bin/gsutil cp -r gs://hotpot-ir/data/hotpot_chunked .
/snap/bin/gsutil cp gs://hotpot-ir/data/derived/wiki-tokenized-bow.tar.bz2 .
/snap/bin/gsutil cp gs://hotpot-ir/data/derived/vocab.json .
/snap/bin/gsutil cp gs://hotpot-ir/data/derived/doc-freq.json .
/snap/bin/gsutil cp gs://hotpot-ir/data/derived/wiki-index.json .
/snap/bin/gsutil cp gs://hotpot-ir/data/derived/wiki-inverted-index.json .
