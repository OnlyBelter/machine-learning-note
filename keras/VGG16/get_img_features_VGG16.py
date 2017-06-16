# https://keras-cn.readthedocs.io/en/latest/other/application/
from keras.models import Sequential
from keras.models import Model
from keras.layers import (Flatten, Dense, Conv2D, MaxPooling2D)
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
import argparse
import json


def vgg16_model(weights_path):
    # this modle totaly has 22 layers with polling 
    model = Sequential()
    # Block 1
    # 这里隐藏了每一层的深度维，本来应该是(3, 3, 3)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', 
                     input_shape=(224, 224, 3), name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    # Block 6, fc
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(1000, activation='softmax', name='predictions'))
    model.load_weights(weights_path)
    return model

def process_pic(img_path, model='', predict=True):
    img_path = img_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # 下面两步不是很理解
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    if predict:  # predict pic's class
        last_layer_features = model.predict(x)  # 1000 last_layer_features
        # print('Predicted:', decode_predictions(last_layer_features, top=3)[0])
        return decode_predictions(last_layer_features, top=3)[0]
    else:  # return 4096 last_layer_features
        last_layer_features = model.predict(x)
        return last_layer_features

def create_data_json(root_d, file_n):
    """
    create dataset.json file and tasks.txt
    :param root_d:
    :param file_n:
    :return:
    """
    a_dic = {'images': [], 'dataset': 'self_img'}
    with open(os.path.join(root_d, file_n)) as f_handle:
        for x in f_handle:
            each_img_dic = {}
            x = x.strip()
            x_list = x.split('\t')
            each_img_dic['filename'] = x_list[0]
            each_img_dic['imgid'] = x_list[1]
            each_img_dic['senences'] = []
            each_img_dic['split'] = 'test'
            each_img_dic['sentids'] = []
            each_img_dic['predict_classes'] = x_list[2:]
            a_dic['images'].append(each_img_dic)
            with open(os.path.join(root_d, 'img', 'tasks.txt'), 'a') as f_handle:
                f_handle.write(x_list[0] + '\n')
    with open(os.path.join(root_d, 'self_img_dataset.json'), 'a') as f_handle:
        f_handle.write(json.dumps(a_dic, indent=2))


def main(params, model):
    # predict images' classes
    self_pic_dir = params['self_pic_dir']
    line = 0
    for root, dirs, files in os.walk(os.path.join(self_pic_dir, 'img')):
        for f in files:
            if f.endswith('jpg'):
                # print(f)
                img_path = os.path.join(root, f)
                predict_results = process_pic(img_path, model=model)
                predict_list = [': '.join(str(i) for i in list(_[1:])) for _ in predict_results]
                output_str = '\t'.join([f, str(line)] + predict_list)
                line += 1
                with open(os.path.join(self_pic_dir, 'predict_images_class.txt'), 'a') as f_handle:
                    f_handle.write(output_str + '\n')
    
    # get images' features
    features = np.zeros([line, 4096])
    line2 = 0
    # 去掉最后一层后，构建一个新的model
    model.layers.pop()
    model2 = Model(model.input, model.layers[-1].output)
    for root, dirs, files in os.walk(os.path.join(self_pic_dir, 'img')):
        for f in files:
            if f.endswith('jpg'):
                print(f)
                # f = 'butterfly1.jpg'
                img_path = os.path.join(root, f)
                features[line2] = process_pic(img_path, model=model2, predict=False)
                line2 += 1
    np.save(os.path.join(self_pic_dir, 'self_img_vgg_feats'), features)
    # features_file = 'self_img_vgg_feats.npy'
    class_file = 'predict_images_class.txt'
    create_data_json(self_pic_dir, class_file)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)
    # 'VGG16 Model pre-training weights in kernels based on tensorflow'
    weights_path = os.path.join(BASE_DIR, 'VGG16_weights', 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    if not os.path.exists(weights_path):
        # this file can be downloaded from https://github.com/fchollet/deep-learning-models/releases
        # or https://pan.baidu.com/s/1dEA0sXb
        print('please download "vgg16_weights_tf_dim_ordering_tf_kernels.h5" and put it into ', weights_path)
    model = vgg16_model(weights_path)
    self_pic_dir = os.path.join(BASE_DIR, 'self_pic')
    parser = argparse.ArgumentParser()
    parser.add_argument('self_pic_dir', default=self_pic_dir,
                        type=str, help='self pictures dir')
    args = parser.parse_args([self_pic_dir])
    params = vars(args)  # convert to ordinary dict
    print('parsed parameters: ')
    print(json.dumps(params, indent=2))
    print('start to predict images\' classes and get VGG features...')
    main(params, model)

