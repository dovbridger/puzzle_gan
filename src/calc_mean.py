
from options.test_options import TestOptions
from data import CreateDataLoader
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.nThreads = 0
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    image_names=[]
    duplicates = False
    # Run the test
    flat_images = []
    for i, data in enumerate(dataset):
        image = data['image']
        image_flat = image.contiguous().view(3, -1)
        flat_images.append(image_flat)
        path = data['path'][0]
        print("image {0}, path: {1}".format(i, path))
        if path in image_names:
            print('duplicate')
            duplicates = True
        image_names.append(path)
    accumulation_vector = torch.cat(flat_images, 1)
    print("Mean: {0}".format(accumulation_vector.mean(dim=1)))
    print("STD: {0}".format(accumulation_vector.std(dim=1)))
    print("Duplicates: {0}".format(duplicates))
