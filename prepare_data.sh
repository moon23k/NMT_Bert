#!/bin/bash
mkdir -p data
cd data

datasets=(wmt_sm iwslt_sm multi30k)
splits=(train valid test)
langs=(en de)
models=(bert albert roberta)


#Download datasets
bash ../scripts/download_sm_datasets.sh

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



for dataset in "${datasets[@]}"; do
    mkdir -p ${dataset}/vocab
    for model in "${models[@]}"; do
        bash ../scripts/build_vocab.sh -i ${dataset}/concat.txt -p ${dataset}/vocab/${model} -m ${model}
    done
    rm ${dataset}/concat.txt
done



#tokens to ids
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        mkdir -p ${dataset}/ids_${model}
        for split in "${splits[@]}"; do
            for lang in "${langs[@]}"; do
                spm_encode --model=${dataset}/vocab/${model}.model --extra_options=bos:eos \
                --output_format=id < ${dataset}/tok/${split}.${lang} > ${dataset}/ids_${model}/${split}.${lang}
            done
        done
    done
done



rm -r sentencepiece

