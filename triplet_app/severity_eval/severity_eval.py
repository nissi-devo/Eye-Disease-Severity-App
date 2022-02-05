from triplet_app import app
import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils import data
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from triplet_app.nets.siamese_net import SiameseNetwork

path_test_csv = os.path.join(app.root_path, "data", "labels_test.csv")
path_test = os.path.join(app.root_path, "data", "masks")
path_anchor_masks = os.path.join(app.root_path, "masks")
test = pd.read_csv(path_test_csv)
test = test.rename(columns={'image': 'Image','level': 'DR_grade'})


def img_processing(input_image):
    '''
    processes PIL image file
    '''
    output_image = input_image.convert('L')
    output_image = output_image.resize((300, 225), Image.ANTIALIAS)

    transf = transforms.Compose([
        transforms.ToTensor()
    ])

    output_image = transf(output_image)
    output_image = np.repeat(output_image, 3, 0)
    output_image = output_image[np.newaxis, ...]
    # output_image = Variable(output_image).cuda()

    return output_image

class SeverityEval:

    @staticmethod
    def load_siamese_net():
        model = SiameseNetwork()
        #model.load_state_dict(torch.load(os.path.join(app.root_path,"nets", "resnet_model.pth")))
        return model

    @staticmethod
    def pooled_test(image_path, imgs, net):

        img_comparison = img_processing(Image.open(path_anchor_masks + "Mask_" + image_path + ".jpg"))

        euclidean_distances = []
        for i in range(len(imgs)):
            img = img_processing(Image.open(path_test + "Mask_" + imgs[i]))
            output0, output1, output2 = net.forward(img_comparison, img, img)
            euclidean_distance = F.pairwise_distance(output0, output1)

            euclidean_distances.append(euclidean_distance.item())

        return np.mean(euclidean_distances)

        # take median euclidean distance compared to the the pool

    # Severity evaluation: anchor vs negative images
    @staticmethod
    def get_negative_samples(anchor_img, label, train):
        max_severity = train.DR_grade.max()
        neg_imgs_list = []
        for i in range(5):
            if label == 0:
                n = train.iloc[np.random.choice(list(train[train.DR_grade == 0 + max_severity].index))]
            else:

                if label == max_severity:
                    n = train.iloc[np.random.choice(list(train[train.DR_grade == 0].index))]
                else:
                    upper_bound = max_severity - label
                    lower_bound = label - 0
                    if upper_bound > lower_bound:
                        n = train.iloc[np.random.choice(list(train[train.DR_grade == label + upper_bound].index))]
                    else:
                        n = train.iloc[np.random.choice(list(train[train.DR_grade == label - lower_bound].index))]

            neg_imgs_list.append(n.Image + '.jpg')

        return {'anchor': anchor_img, 'neg_imgs': neg_imgs_list}

    def get_single_severity(self, img, label):
        net = SeverityEval.load_siamese_net()
        result = SeverityEval.get_negative_samples(img, label, test)
        distance = SeverityEval.pooled_test(result['anchor'].split(".")[0], result['neg_imgs'], net)

        return distance

