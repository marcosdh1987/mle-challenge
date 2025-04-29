#!/bin/bash
DIR=datasets/kaggle_dataset2/
mkdir -p "$DIR"
wget -P "$DIR" https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip
cd "$DIR"
unzip -q dataset-resized.zip
# Dividir en train, validation y test (70/15/15)
for class_dir in dataset-resized/*/; do
    class_name=$(basename "$class_dir")
    mkdir -p train/"$class_name" validation/"$class_name" test/"$class_name"
    files=("${class_dir}"* )
    total=${#files[@]}
    num_train=$(( total * 70 / 100 ))
    num_val=$(( total * 15 / 100 ))
    # Mezclar archivos en Bash con RANDOM
    for ((i=total-1; i>0; i--)); do
        j=$((RANDOM % (i+1)))
        tmp=${files[i]}
        files[i]=${files[j]}
        files[j]=$tmp
    done
    for idx in "${!files[@]}"; do
        src_file="${files[$idx]}"
        if [ "$idx" -lt "$num_train" ]; then
            mv "$src_file" train/"$class_name"/
        elif [ "$idx" -lt $(( num_train + num_val )) ]; then
            mv "$src_file" validation/"$class_name"/
        else
            mv "$src_file" test/"$class_name"/
        fi
    done
done
# Limpiar archivos temporales
echo "Limpiando artefactos temporales..."
rm -rf dataset-resized dataset-resized.zip __MACOSX

echo "Dataset split en train/, validation/ y test/"
