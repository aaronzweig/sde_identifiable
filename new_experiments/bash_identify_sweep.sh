for seed in 42 43 44 45 46; do
    python updated_identifiability_script.py --seed $seed --k 3
done
for seed in 42 43 44 45 46; do
    python updated_identifiability_script.py --seed $seed --k 2
done