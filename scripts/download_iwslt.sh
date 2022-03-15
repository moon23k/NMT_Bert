#!bin/bash

mkdir -p iwslt
cd iwslt
mkdir -p raw

wget "http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
tar zxvf de-en.tgz
rm de-en.tgz

langs=(en de)

for l in "${langs[@]}"; do
    f_in=de-en/train.tags.de-en.$l
    f_out=de-en/train_orig.$l

    cat $f_in \
    | grep -v '<url>' \
    | grep -v '<talkid>' \
    | grep -v '<keywords>' \
    | grep -v '<speaker>' \
    | grep -v '<reviewer' \
    | grep -v '<translator' \
    | grep -v '<doc' \
    | grep -v '</doc>' \
    | sed -e 's/<title>//g' \
    | sed -e 's/<\/title>//g' \
    | sed -e 's/<description>//g' \
    | sed -e 's/<\/description>//g' \
    | sed 's/^\s*//g' \
    | sed 's/\s*$//g' \
    > $f_out
done


for l in "${langs[@]}"; do
    for o in `ls de-en/IWSLT*.TED*.de-en.$l.xml`; do
        fname=${o##*/}
        f=de-en/${fname%.*}
        grep '<seg id' $o \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\Ã¢â‚¬â„¢/\'/g" \
        > $f
        rm $o
    done
done   


for l in "${langs[@]}"; do
    awk '{if (NR%23 == 0)  print $0; }' de-en/train_orig.$l > raw/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' de-en/train_orig.$l > raw/train.$l

    cat de-en/IWSLT14.TED.dev2010.de-en.$l \
        de-en/IWSLT14.TEDX.dev2012.de-en.$l \
        de-en/IWSLT14.TED.tst2010.de-en.$l \
        de-en/IWSLT14.TED.tst2011.de-en.$l \
        de-en/IWSLT14.TED.tst2012.de-en.$l \
        > raw/test.$l
done

rm -r de-en

echo '--- use only 30,000 rows for training dataset ---'
for l in "${langs[@]}"; do
    sed -n '1, 30000p' raw/train.$l > raw/tmp_train.$l
    rm raw/train.$l
    mv raw/tmp_train.$l raw/train.$l
done


echo '--- use only 1,000 rows for valid and test dataset ---'
vt=(valid test)
for split in "${vt[@]}"; do
    for l in "${langs[@]}"; do
        sed -n '1, 1000p' raw/$split.$l > raw/tmp_$split.$l
        rm raw/$split.$l
        mv raw/tmp_$split.$l raw/$split.$l
    done
done

cd ..