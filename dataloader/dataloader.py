from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from util.util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.sketch_path = join(image_dir, "a")
        self.photo_path = join(image_dir, "b")
        self.sketch_filenames = [x for x in listdir(self.sketch_path) if is_image_file(x)]

        # print self.sketch_filenames
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image

        input = load_img(join(self.sketch_path, self.sketch_filenames[index])).convert('L')
        input = self.transform(input)

        target_filename = self.sketch_filenames[index]
        # target=load_img(join(self.photo_path, target_filename)).convert('L')
        # target = self.transform(target)
        if (self.sketch_filenames[index].startswith('f') or self.sketch_filenames[index].startswith('s')):
            target = load_img(join(self.photo_path, self.sketch_filenames[index])).convert('L')
            target = self.transform(target)
        else:
            target_filename = self.sketch_filenames[index].split('_')[0] + '_' + \
                              self.sketch_filenames[index].split('_')[-1]
            target = load_img(join(self.photo_path, target_filename)).convert('L')
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.sketch_filenames)


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir)

if __name__ == "__main__":
    get_training_set("data")
    get_test_set("data")
    print("done")