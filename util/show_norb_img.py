import os
import torch
from PIL import Image

def concat_images(data, labels, num, name):
    w, h = 96, 96
    inst_num = 5 # 5 instances in total
    concat = Image.new('RGB', (w*num, h*inst_num))
    cnt = {}
    finished = 0
    for i, label in enumerate(labels):
        img = data[2*i]
        if label not in cnt:
            cnt[label] = 0
        elif cnt[label] < num:
            cnt[label] += 1
        elif cnt[label] == num:
            finished += 1
            if finished == inst_num:
                break
        else:
            continue
        h_, w_ = img.shape
        assert (h_, w_) == (h, w), "({}, {}) vs ({}, {})".format(h_, w_, h, w)
        img = Image.fromarray(img.numpy(), mode='L')
        offset = (cnt[label]*w, label*h)
        concat.paste(img, offset)
    print('Concatenating {} images into {}'.format(num*inst_num, name))
    concat.save(name)


def show_image(data, labels, info, num=10):
    print('Close image and press ENTER to view next ...')
    for i in range(num):
        img, label = data[i], labels[int(i/2)]
        instance, elevation, azimuth, lighting = info[int(i/2)]
        img = Image.fromarray(img.numpy(), mode='L')
        title = 'label_{}_instance_{}_elevation_{}_azimuth_{}_lighting_{}'\
                .format(label, instance, elevation, azimuth, lighting)
        print(title)
        img.show(title)
        input()


if __name__ == '__main__':
    root = './data/smallNORB'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    train_data, train_labels, train_info = torch.load(
        os.path.join(root, processed_folder, training_file))

    test_data, test_labels, test_info = torch.load(
        os.path.join(root, processed_folder, test_file))
    
    # generate batch image, each column is an instance
    concat_images(train_data, train_labels, 5, 'smallNORB-train.jpg')
    concat_images(test_data, test_labels, 5, 'smallNORB-test.jpg')

    # view image one-by-one
    # show_image(train_data, train_labels, train_info)
