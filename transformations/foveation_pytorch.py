import numpy as np
from PIL import Image

from retina_transform import foveat_img

class Foveate(object):

    def __init__(self, crop_size=None, p_val=None):
        self.crop_size=crop_size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be foveated.
        Returns:
            PIL Image: Foveated image.
        """
        return Image.fromarray(foveat_img(np.asarray(img), [(self.crop_size / 2, self.crop_size / 2)]))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
                                                                         
