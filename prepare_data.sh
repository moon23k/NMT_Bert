#!/bin/bash
mkdir -p data
cd data

datasets=(multi30k iwslt wmt)
splits=(train valid test)
langs=(en de)



#Download datasets
for dataset in "${datasets[@]}"; do
    bash ../scripts/download_${dataset}.sh
done



#Peer Tokenize with sacremoses
python3 -m pip install -U sacremoses
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for lang in "${langs[@]}"; do
            mkdir -p ${dataset}/tok
            sacremoses -l ${lang} -j 8 tokenize < ${dataset}/raw/${split}.${lang} > ${dataset}/tok/${split}.${lang}
        done
    done
done



#concat all files into one file to build vocab
for dataset in "${datasets[@]}"; do
    cat ${dataset}/tok/* >> ${dataset}/concat.txt
done



#Build vocab
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig
cd ../../


vocabs=(vocab albert_vocab roberta_vocab)

for dataset in "${datasets[@]}"; do
    mkdir -p ${dataset}/vocab
    for vocab in "${vocabs[@]}"; do
        bash ../scripts/build_vocab.sh -i ${dataset}/concat.txt -p ${dataset}/vocab/${vocab} -c ${vocab}
    done
    rm ${dataset}/concat.txt
done


declare -A ids
ids[vocab]=ids
ids[albert_vocab]=ids_albert
ids[roberta_vocab]=ids_roberta


#tokens to ids
for dataset in "${datasets[@]}"; do
    for vocab in "${vocabs[@]}"; do
        mkdir -p ${dataset}/${ids[${vocab}]}
        for split in "${splits[@]}"; do
            for lang in "${langs[@]}"; do
                spm_encode --model=${dataset}/vocab/${vocab}.model --extra_options=bos:eos \
                --output_format=id < ${dataset}/tok/${split}.${lang} > ${dataset}/${ids[${vocab}]}/${split}.${lang}
            done
        done
    done
done



rm -r sentencepiece

