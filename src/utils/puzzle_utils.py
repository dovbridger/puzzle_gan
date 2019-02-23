import os

def create_test_completions(input_dir, experiment_name='inpaint_test'):
    for folder in os.listdir(input_dir):
        command = 'python test.py --model inpainting --phase {0} --name {1} --dataroot {2}'.format(folder, experiment_name, input_dir)
        command += ' --how_many 1000000 --burn_extent 2  --generator_window 4 --save_images_frequency 1 --which_epoch 32'
        os.system(command)
