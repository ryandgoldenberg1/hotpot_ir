
# Download Hotpot Data
wget -c -P hotpot/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget -c -P hotpot/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
wget -c -P hotpot/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget -c -P hotpot/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json

# Download GloVe
GLOVE_DIR=hotpot/
mkdir -p $GLOVE_DIR
wget -c http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip -P hotpot/ $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

# Download Spacy language models
python3 -m spacy download en
