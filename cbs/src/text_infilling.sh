# model_names="LLAMA-3.1-8B-Instruct LLAMA-3.1-8B XLMR-Large GEMMA-3-12B-IT SOLAR-10.7B-Instruct SOLAR-10.7B-v1.0 EXAONE-3.5-7.8B-Instruct EXAONE-Deep-7.8B-AWQ"
model_names="Trillion-7B-preview"
languages="ko en"
batch_size=32

for model_name in $model_names; do
    for language in $languages; do
        CUDA_VISIBLE_DEVICES=0 python3 text_infilling.py \
            --model_name $model_name \
            --language $language \
            --check_prev
    done
    python3 ../results/text_infilling/evaluate.py \
        --model_name $model_name \
        --compare \
        --draw_figure
done