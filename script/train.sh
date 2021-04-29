# train VAE & GAN and doing encoding at the same time
sudo /home/ubuntu/anaconda3/envs/dgl/bin/python main.py --train --validation --test --save_result --validation_interval 1000 --log_interval 100 --max_step 70000 --dataset_root /home/ubuntu/data/BottleImg --num_resblock 12 --latent_dim 256 --use_ref

# encode lc and shape first, then VAE & GAN
sudo /home/ubuntu/anaconda3/envs/dgl/bin/python main.py --train --validation --test --save_result --validation_interval 1000 --log_interval 100 --max_step 60000 --dataset_root /home/ubuntu/data/BottleImg --num_resblock 12 --finetune --finetune_step 30000 --latent_dim 256 --encoder vae --encoder vae --use_ref

# style-gan encoder
sudo /home/menjarleev/anaconda3/envs/eecs545/bin/python main.py --train --validation --test --save_result --validation_interval 1000 --log_interval 100 --max_step 90000 --dataset_root /home/menjarleev/developer/dataset/BottleImg --num_resblock 12 --finetune_step 50000 --latent_dim 256 --encoder style-gan


# inference
main.py --inference --model_dir ./ckpt/photometricGAN_1  --num_resblock 12 --dataset_root ./


sudo /home/menjarleev/anaconda3/envs/eecs545/bin/python main.py --train --validation --test --save_result --validation_interval 1000 --log_interval 100 --max_step 60000 --dataset_root /home/menjarleev/developer/dataset/BottleImg --num_resblock 12 --encoder style-gan --skip_connect
sudo /home/menjarleev/anaconda3/envs/eecs545/bin/python main.py  --train --validation --test --save_result --validation_interval 5000 --log_interval 100 --max_step 90000 --dataset_root /home/menjarleev/developer/dataset/BottleImg --num_resblock 12 --finetune --finetune_step 60000 --latent_dim 256 --encoder vae --use_ref