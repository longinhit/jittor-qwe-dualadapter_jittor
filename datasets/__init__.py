from .jittor_dataset import JittorDataSet

dataset_list = {
                'jittor_dataset': JittorDataSet,
                }


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)