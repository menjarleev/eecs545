$lr_array = @("0.003", "0.001", "0.0003", "0.0001", "0.00003", "0.00001")
#$max_step_array = @("500", "1000", "1500", "2000", "3000", "4000", "5000", "7000")
$max_step_array = @("500", "600")
#$max_step_array = @("2", "3", "4", "5", "6")

$nres_array = @("4","5","6","7","8","9","10","11","12","15")


#for ($num_resblock = 5; $num_resblock -le 8; $num_resblock++ )
foreach ($lr in $lr_array)
{
    foreach ($max_step in $max_step_array)
    {
        foreach ($num_resblock in $nres_array)
        {
            $foldername = [string]::Format("lr{0}maxstep{1}nres{2}",$lr,$max_step,$num_resblock)
            py -3.8  .\main.py --test --test_step 50 --dataset_root .\dataset --num_resblock $num_resblock --train --max_step $max_step --validation_interval 100 --log_interval 100 --lr $lr --validation --optim_name Adam --save_result --gpu_id 0 --ckpt_root ./ckpt/$foldername
            
        }

    }
    
}

#py -3.8  .\main.py --test --test_step 50 --dataset_root .\dataset --num_resblock 12 --train --max_step 5 --validation_interval 100 --log_interval 100 --lr 0.00001 --validation --optim_name Adam --save_result --gpu_id 0 --ckpt_root ./ckpt/test1

#py -3.8  .\main.py --test --test_step 50 --dataset_root .\dataset --num_resblock 12 --train --max_step 5 --validation_interval 100 --log_interval 100 --lr 0.00001 --validation --optim_name Adam --save_result --gpu_id 0 --ckpt_root ./ckpt/test2