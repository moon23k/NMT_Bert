#!bin/bash
git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset

mkdir -p multi30k 
mv multi30k-dataset/data/task1/raw multi30k
rm -r multi30k-dataset


cd multi30k/raw
ls | grep -P '^((?!en|de).)*$' | xargs rm
ls | grep -P "2017" | xargs rm 
ls | grep -P "2018" | xargs rm


gzip -d train* val* test*

mv val.en valid.en
mv val.de valid.de
mv test_2016_flickr.en test.en
mv test_2016_flickr.de test.de

cd ..