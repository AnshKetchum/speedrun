# train model
accelerate launch alg/run.py --model_name "kaizen9/test2b12" --output_dir test_modelnewLR

# eval model
python eval.py --model_name test_model/checkpoint-16000

python eval.py --model_name test_model/checkpoint-16000 --tasks hellaswag

python eval.py --model_name test_model/checkpoint-16000 --tasks arc_challenge

