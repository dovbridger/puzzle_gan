start "visdom server" python -m visdom.server -port 1986
python train.py --input_nc 1 --output_nc 1 --name toy_example_e1 --dataroot ../datasets/mnist