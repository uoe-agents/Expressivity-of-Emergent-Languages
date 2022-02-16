# list random seeds below
for s in 123 1234 12345
do
python same_complex_diff_pred.py --contrastive --lr 0.0001 --vocab_size 10 --max_len 6 --generalisation_path ./log_scdp/ --language_save_freq 100000 --random_seed $s
done

