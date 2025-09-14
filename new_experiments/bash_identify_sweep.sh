for seed in 1 2 3 4 5 6 7 8 9 10; do
    python updated_identifiability_script.py --seed $seed --k 3 --hidden_dim 20
done
for seed in 1 2 3 4 5 6 7 8 9 10; do
    python updated_identifiability_script.py --seed $seed --k 2 --hidden_dim 20
done