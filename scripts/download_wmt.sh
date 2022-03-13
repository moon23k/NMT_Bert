#!/bin/bash
mkdir -p wmt
cd wmt

#Getting Europarl data for training
wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
tar zxvf training-parallel-europarl-v7.tgz
cd training && ls | grep -v 'europarl-v7.de-en.[de,en]' | xargs rm
cd .. && mv training/europarl* . && rm -r training training-parallel-europarl-v7.tgz


#Getting newstest2014 data for validation
wget --trust-server-names http://www.statmt.org/wmt14/test-filtered.tgz

#Getting newstest2017 data for testing
wget --trust-server-names http://data.statmt.org/wmt17/translation-task/test.tgz

tar zxvf test-filtered.tgz && tar zxvf test.tgz
cd test && ls | grep -v '.*deen\|.*ende' | xargs rm
cd .. && mv test/* . && rm -r test-filtered.tgz test.tgz test


#Download input-from-sgm.perl file (for processing sgm files)
wget -nc https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/ems/support/input-from-sgm.perl


mkdir -p raw
euro=europarl-v7.de-en
validset=newstest2014-deen
testset=newstest2017-ende
langs=(en de)

#use only 30,000 rows for training dataset
for l in "${langs[@]}"; do
    sed -n '1, 30000p' $euro.$l > raw/train.$l
done


perl input-from-sgm.perl < $validset-src.en.sgm > raw/valid.en
perl input-from-sgm.perl < $validset-ref.de.sgm > raw/valid.de
perl input-from-sgm.perl < $testset-src.en.sgm > raw/test.en
perl input-from-sgm.perl < $testset-ref.de.sgm > raw/test.de

ls | grep "$euro." | xargs rm
ls | grep ".sgm" | xargs rm
ls | grep ".perl" | xargs rm


#use only 1,000 rows for valid and test dataset
for split in "${splits[@]}"; do
    for lang in "${langs]@}"; do
        sed -n '1, 1000p' raw/$split.$lang > raw/tmp_$split.$lang
        rm raw/$split.$lang
        mv -i raw/tmp_$split.$lang raw/$split.$lang
    done
done

cd ..