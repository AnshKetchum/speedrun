# train model
# accelerate launch alg/run.py --model_name "kaizen9/test2b12" --output_dir test_modelnewLR

# train universal transformer

# --ut_n_blocks = number of blocks
# --ut_max_steps = loops per block
# --ut_beta     = entropy regularization coefficient β for Ouro ELBO

mkdir -p logs

accelerate launch alg/run.py \
    --model_type ouro \
    --ut_tokenizer gpt2 \
    --ut_d_model 128 \
    --ut_n_heads 4 \
    --ut_d_ff 128 \
    --ut_max_steps 6 \
    --ut_beta 0.01 \
    --ut_dropout 0.1 \
    --no_distil \
    --max_steps 500 \
    --save_steps 10 \
    --torch_compile false \
    --output_dir checkpoints/ouro_model_6_steps_beta_001 \
    --run_name ouro_beta_001 \
    2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log

# accelerate launch alg/run.py \
#     --model_type universal_transformer \
#     --ut_tokenizer gpt2 \
#     --ut_d_model 128 \
#     --ut_n_heads 4 \
#     --ut_d_ff 256 \
#     --ut_max_steps 6 \
#     --ut_n_blocks 1 \
#     --ut_eps 0.01 \
#     --ut_tau 0.0001 \
#     --ut_dropout 0.1 \
#     --no_distil \
#     --max_steps 500 \
#     --save_steps 10 \
#     --output_dir checkpoints/ut_model_1_block_6_loop_tau_0_0001 \
#     --run_name universal_transformer_tau_0001 \
#     --visualize_angular_distances viz/ut_model_1_block_6_loop_tau_0_0001 \
#     2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
