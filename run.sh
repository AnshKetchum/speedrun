# train model
# accelerate launch alg/run.py --model_name "kaizen9/test2b12" --output_dir test_modelnewLR

# train universal transformer

# --ut_n_blocks = number of blocks 
# --ut_max_steps = loops per block
accelerate launch alg/run.py \
    --model_type universal_transformer \
    --ut_tokenizer gpt2 \
    --ut_d_model 128 \
    --ut_n_heads 4 \
    --ut_d_ff 256 \
    --ut_max_steps 6 \
    --ut_n_blocks 1 \
    --ut_eps 0.01 \
    --ut_tau 0.01 \
    --ut_dropout 0.1 \
    --no_distil \
    --max_steps 1000 \
    --save_steps 150 \
    --output_dir ut_model \
    --run_name universal_transformer
