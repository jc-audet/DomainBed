
for i in {0..20}
do
    python3 -m domainbed.scripts.train\
        --algorithm ANDMask\
        --dataset Spirals\
        --test_env 0 \
        --holdout_fraction 0.001\
        --steps 600 \
        --seed $i
done