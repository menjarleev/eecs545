# PhotometricGAN
## run code
```bash
main.py --test --test_step 50 --dataset_root YOUR_DATASET_ROOT --num_resblock 12 
--train --max_step 500 --validation_interval 100 --log_interval 100 
--lr 0.00001 --validation --optim_name Adam --save_result --batch_size 8 --batch_size_eval 8
```
Args can be found in ```option/option.py```

## configuration
* TODO add configuration files