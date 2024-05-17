import yaml

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __getattr__(self, name):
        return self.__dict__.get(name)

def load_config(file_path):
    with open(file_path) as config_file:
        config_dict = yaml.safe_load(config_file)
    return Config(**config_dict)

# config = load_config('config.yaml')
#
# # Now you can access config.patch_size directly
# print(config.patch_size)
