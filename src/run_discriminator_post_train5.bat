python train.py --phase True --model post_gan --name batch1_burn4_dw64_post_40 --fake_loss_weight 0.5 --burn_extent 4 --dataroot ../../puzzle_gan_data/datasets/puzzle_parts/validation --niter 40 --niter_decay 0 --generator_window 8 --discriminator_window 64 --display_id -1 --save_epoch_freq 5 --batchSize 1 --discriminator_to_load batch1_burn4_dw64 --discriminator_epoch 40