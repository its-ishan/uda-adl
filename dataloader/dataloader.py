from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from util.util import is_image_file, load_img
from torch.utils.data import DataLoader


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.sketch_path = join(image_dir, "a")  # 11k images
        self.photo_path = join(image_dir, "b")
        self.sketch_filenames = [x for x in listdir(self.sketch_path) if is_image_file(x)]

        # print self.sketch_filenames
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image

        input = load_img(join(self.sketch_path, self.sketch_filenames[index])).convert(
            'L')  # sketch is a photo is b gan
        #print(f'input file name: {(join(self.sketch_path, self.sketch_filenames[index]))}')
        input = self.transform(input)
        target_filename = self.sketch_filenames[index]
        # target=load_img(join(self.photo_path, target_filename)).convert('L')
        # target = self.transform(target)
        if (self.sketch_filenames[index].startswith('f') or self.sketch_filenames[index].startswith('s')):
            #print("inside if")
            #print(f' target filename : {join(self.photo_path, self.sketch_filenames[index])}')
            target = load_img(join(self.photo_path, self.sketch_filenames[index])).convert('L')
            target = self.transform(target)
        else:
            target_filename = self.sketch_filenames[index].split('_')[0] + '_' + \
                              self.sketch_filenames[index].split('_')[-1]
            #print("inside else")
            target = load_img(join(self.photo_path, target_filename)).convert('L')
            #print(f' target filename : {join(self.photo_path, target_filename)}')
            target = self.transform(target)
        return input, target

    def __len__(self):
        return len(self.sketch_filenames)
class DatasetFromFolder2(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder2, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.a_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.b_filenames = [x for x in listdir(self.b_path) if is_image_file(x)]

        # print self.sketch_filenames
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a_filename_parts = self.a_filenames[index].split('_')
        if "actual" in a_filename_parts:
            b_filename = a_filename_parts[0] + '_' + a_filename_parts[-1]
            input_a = load_img(join(self.a_path, self.a_filenames[index])).convert('L')
            input_b = load_img(join(self.b_path, b_filename)).convert('L')
            #print(f' input_a : {self.a_filenames[index]} input_b : {b_filename}')
            input_a = self.transform(input_a)
            input_b = self.transform(input_b)
            return input_a, input_b
        else:
            input_a = load_img(join(self.a_path, self.a_filenames[index])).convert('L')
            input_b = load_img(join(self.b_path, self.b_filenames[index])).convert('L')
            input_a = self.transform(input_a)
            input_b = self.transform(input_b)
            return input_a, input_b
    def __len__(self):
        return len(self.a_filenames)


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return data.default_collate(batch)

def get_training_set(root_dir):
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir)

if __name__ == "__main__":
    print(type(get_training_set("/mnt/nvme0n1p5/projects/UDA-ADL/data")))
    train_set = get_training_set("/mnt/nvme0n1p5/projects/UDA-ADL/data")
    training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=10,
                                      shuffle=True)

    print(f'iteration: {len(training_data_loader)}')
    for iteration, batch in enumerate(training_data_loader):
        if batch is not None:
            print("iteration: ", iteration)
            print("batch_length: ", len(batch))
            print(f'batch[0] : {batch[0].shape} batch[1] : {batch[1].shape}')

    #get_test_set("/mnt/nvme0n1p5/projects/UDA-ADL/data")
    print("done")