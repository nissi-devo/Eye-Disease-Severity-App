from PIL import  Image
from triplet_app import app
import numpy as np
import os


filepath = '/uploads/'
image_size = 224

class DataPreProcessor:
    def read_resize(self, filename):
        im = Image.open(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)).convert('RGB')
        im = im.resize((image_size, image_size))
        return np.array(im, dtype="float32")

