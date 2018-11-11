#! /bin/bash

sudo apt update
sudo apt -y upgrade
sudo apt install -y python3-pip

pip3 install spacy
python3 -m spacy download en
