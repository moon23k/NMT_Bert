#!/bin/bash
mkdir -p data
cd data

splits=(train valid test)
langs=(en de)
models=(bert albert roberta)


#download small wmt dataset
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11ln6BiC4l1kk-vCvKGgQngSzoDOai4iP' -O wmt_sm.tar.gz
tar -zxvf wmt_sm.tar.gz
rm wmt_sm.tar.gz


#move original files to raw folder
cd wmt_sm
mkdir -p raw
mv `ls | grep -v raw` raw
cd ..


#Peer Tokenize with sacremoses
python3 -m pip install -U sacremoses
for split in "${splits[@]}"; do
    for lang in "${langs[@]}"; do
        mkdir -p wmt_sm/tok
        sacremoses -l ${lang} -j 8 tokenize < wmt_sm/raw/${split}.${lang} > wmt_sm/tok/${split}.${lang}
    done
done


#concat all files into one file to build vocab
cat wmt_sm/tok/* >> wmt_sm/concat.txt


#Get Sentencepiece
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig
cd ../../


#Build Vocab with Sentencepice
mkdir -p wmt_sm/vocab
for model in "${models[@]}"; do
    bash ../scripts/build_vocab.sh -i wmt_sm/concat.txt -p wmt_sm/vocab/${model} -m ${model}
done
rm wmt_sm/concat.txt



#convert tokens to ids
for model in "${models[@]}"; do
    mkdir -p wmt_sm/ids_${model}
    for split in "${splits[@]}"; do
        for lang in "${langs[@]}"; do
            spm_encode --model=wmt_sm/vocab/${model}.model --extra_options=bos:eos \
            --output_format=id < wmt_sm/tok/${split}.${lang} > wmt_sm/ids_${model}/${split}.${lang}
        done
    done
done


rm -r sentencepiece