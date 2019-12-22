import glob

import os


def create_images_list(path_root):
    result = []
    image_dict = {}
    for f in glob.glob("{}/*/*.jpg".format(path_root)):
        label_name = os.path.basename(os.path.dirname(f))
        file_name = os.path.basename(f)
        if label_name not in image_dict:
            image_dict[label_name] = float(len(image_dict))

        result.append([image_dict[label_name], os.path.join(label_name, file_name)])

    return result, image_dict


def get_labels(list_file):
    labels = set()
    with open(list_file, "r") as f:
        for l in f.readlines():
            line_parts = l.split("\t")
            label_name = line_parts[1]
            labels.add(label_name)

    return labels
