model_names="LLAMA-3.1-8B-Instruct GEMMA-3-12B-IT SOLAR-10.7B-Instruct EXAONE-3.5-7.8B-Instruct EXAONE-Deep-7.8B-AWQ"
languages="ko en"

for model_name in $model_names; do
    for language in $languages; do
        CUDA_VISIBLE_DEVICES=1 python3 extractive_QA.py \
            --model_name $model_name \
            --language $language
    done
done