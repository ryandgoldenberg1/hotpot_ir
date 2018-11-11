#! /bin/bash

set -e

function usage() {
  echo "Usage: tokenize.sh <subdir>"
  exit 1
}

if [ -z "$1" ]; then
  usage
fi

/snap/bin/gsutil cp gs://hotpot-ir/data/derived/wiki-parsed.tar.bz2 .
tar -xvf wiki-parsed.tar.bz2
inputPath="wiki-parsed/$1"
if [ ! -d "$inputPath" ]; then
  echo "Path not found: $inputPath"
  exit 1
fi

outputPath="wiki-tokenized/$1"
echo "Using input path [$inputPath], outputPath [$outputPath]"
python3 hotpot_ir/tokenizer.py -i $inputPath -o $outputPath -t 2

storagePath="gs://hotpot-ir/data/derived/$outputPath"
echo "Finished tokenizing documents, copying them to $storagePath"
/snap/bin/gsutil cp -r $outputPath $storagePath
