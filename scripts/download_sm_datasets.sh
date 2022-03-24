#!/bin/bash

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11ln6BiC4l1kk-vCvKGgQngSzoDOai4iP' -O wmt_sm.tar.gz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11ifvX0goWvUjadMcnnMX7FsySY1VAQOj' -O iwslt_sm.tar.gz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11PpkMxJJ99euSoR1PwZVc15hM-CAmLds' -O multi30k.tar.gz


tar -zxvf wmt_sm.tar.gz
tar -zxvf iwslt_sm.tar.gz
tar -zxvf multi30k.tar.gz

rm *gz


datasets=(wmt_sm iwslt_sm multi30k)

for data in "${datasets[@]}"; do
    cd $data/
    mkdir -p raw
    mv train* valid* test* raw/
    cd ..
done