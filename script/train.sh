# train VAE & GAN and doing encoding at the same time
sudo /home/ubuntu/anaconda3/envs/dgl/bin/python main.py --train --validation --test --save_result --validation_interval 1000 --log_interval 100 --max_step 70000 --dataset_root /home/ubuntu/data/BottleImg --num_resblock 12

# encode lc and shape first, then VAE & GAN
sudo /home/ubuntu/anaconda3/envs/dgl/bin/python main.py --train --validation --test --save_result --validation_interval 1000 --log_interval 100 --max_step 70000 --dataset_root /home/ubuntu/data/BottleImg --num_resblock 12 --finetune --finetune_step 30000
