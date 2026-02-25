#!/bin/sh
for seed in $(seq 1 10); do
    python updated_identifiability_script.py --seed $seed --k 4 --c_scale 0.5 --lr 0.002 --hidden_dim 20
done