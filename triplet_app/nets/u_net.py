from triplet_app import app
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

class UNet:
    def read_image(self, img_name):
        h = 512
        w = 512
        x = cv2.imread(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], img_name), cv2.IMREAD_COLOR)
        x = cv2.resize(x, (w, h))
        ori_x = x
        x = x / 255.0
        x = x.astype(np.float32)
        return ori_x, x

    def load_unet(self):
        model = load_model(os.path.join(app.root_path,"nets", "model.h5"), custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou': iou})
        return model

    def get_predicted_mask(self, img_name):
        ori_x,x = self.read_image(img_name)
        mask_path = os.path.join(app.root_path, app.config['MASK_UPLOADER'], f"Mask_{img_name}")
        model = self.load_unet()
        y = model.predict(np.expand_dims(x, axis=0))[0]
        y = y > 0.5
        y = y.astype(np.int32)
        y = np.squeeze(y, axis=-1)
        y = np.expand_dims(y, axis=-1)
        y = np.concatenate([y, y, y], axis=-1) * 255

        cv2.imwrite(mask_path, y)