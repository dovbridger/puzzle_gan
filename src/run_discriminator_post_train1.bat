python train.py --phase validation --model post_gan_virtual --name virtual_test_post_8 --fake_loss_weight 0.5 --burn_extent 2 --dataroot ../../puzzle_gan_data/datasets/virtual_puzzle_parts --niter 40 --niter_decay 0 --generator_window 4 --discriminator_window 64 --display_id 1 --save_epoch_freq 5 --batchSize 1 --discriminator_to_load virtual_test --discriminator_epoch 8