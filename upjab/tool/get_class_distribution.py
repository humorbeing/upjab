def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    idx2class = {v: k for k, v in dataset_obj.class_to_idx.items()}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict


if __name__ == '__main__':
    import torchvision

    dataset_root = 'example_data/text/long-tail_dataset'

    def whatis(x):
        return 0
    dataset = torchvision.datasets.DatasetFolder(root=dataset_root, loader=whatis, extensions=['.txt'], transform=None)

    print("Distribution of classes: \n", get_class_distribution(dataset))

    print(dataset.classes)
    print(dataset.class_to_idx)

    print('done')

