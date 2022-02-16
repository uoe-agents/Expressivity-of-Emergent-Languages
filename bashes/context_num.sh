# list random seeds below
for s in 123 1234 12345
do
python context_num.py --contrastive --lr 0.0001 --vocab_size 10 --max_len 6 --generalisation_path ./log/ --language_save_freq 20 --random_seed $s
done