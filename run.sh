# train model
# accelerate launch alg/run.py --model_name "kaizen9/test2b12" --output_dir test_modelnewLR

# train universal transformer

# --ut_n_blocks = number of blocks 
# --ut_max_steps = loops per block
mkdir -p logs
accelerate launch alg/run.py \
    --model_type universal_transformer \
    --ut_tokenizer gpt2 \
    --ut_d_model 128 \
    --ut_n_heads 4 \
    --ut_d_ff 256 \
    --ut_max_steps 2 \
    --ut_n_blocks 3 \
    --ut_eps 0.01 \
    --ut_tau 0.01 \
    --ut_dropout 0.1 \
    --no_distil \
    --max_steps 1000 \
    --save_steps 150 \
    --output_dir checkpoints/ut_model_3_block_2_loop \
    --run_name universal_transformer \
    2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
