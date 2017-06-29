#!/usr/bin/bash

### https://github.com/adbar/vardial-experiments
### Adrien Barbaresi, 2017.
### GNU GPL v3 license

### requires SoMaJo tokenizer (https://pypi.python.org/pypi/SoMaJo)


rm -r train/ test/

for lang in bs es-AR es-ES es-PE fa-AF fa-IR fr-CA fr-FR hr id my pt-BR pt-PT sr
    do

    ## TRAINING

    mkdir -p train/$lang

    # DSL-DEV.txt
    cat DSL-TRAIN.txt | grep -P "\t$lang$" | perl -pe "s/\t[a-zA-Z-]{2,5}$//g" | tokenizer --parallel 4 --paragraph_separator single_newlines - | perl -pe 's/\n/ /' | perl -pe 's/\s{2,}/\n/g' > train/$lang/$lang.train

    cd train/$lang/
    split $lang.train --suffix-length=4 --lines=1
    rm $lang.train

    cd ../..



    ## GOLD

    #mkdir -p test/$lang
    #grep -P "\t$lang$" DSL-test.txt | perl -pe "s/\t[a-zA-Z-]{2,5}$//g" | tokenizer --parallel 4 --paragraph_separator single_newlines - | perl -pe 's/\n/ /' | perl -pe 's/\s{2,}/\n/g' > test/$lang/$lang.test

    #cd test/$lang/
    #split $lang.test --suffix-length=4 --lines=1
    #rm $lang.test

    #cd ../..

    done


## TEST

mkdir -p test/test/
cat DSL-test.txt | tokenizer --parallel 4 --paragraph_separator single_newlines - | perl -pe 's/\n/ /' | perl -pe 's/\s{2,}/\n/g' > test/test/data.test

cd test/test/
split data.test --suffix-length=4 --lines=1
rm data.test

cd ../..
