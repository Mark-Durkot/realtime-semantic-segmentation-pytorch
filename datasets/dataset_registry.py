dataset_hub = {}


def register_dataset(dataset_class):
    dataset_hub[dataset_class.__name__.lower()] = dataset_class
    print(f"Registered dataset: {dataset_class.__name__.lower()}")
    return dataset_class