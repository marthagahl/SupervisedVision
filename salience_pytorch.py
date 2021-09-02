import numpy as np
from scipy import ndimage as ndi

from skimage._shared.utils import (get_bound_method_class, safe_as_int, warn,
                             convert_to_float)


from skimage.transform._geometric import (SimilarityTransform, AffineTransform,
                         ProjectiveTransform, _to_ndimage_mode)
from skimage.transform._warps_cy import _warp_fast
from skimage.measure import block_reduce


class SalienceSampling(object):

    def __init__(self, path, num_points, crop_size):
        self.path=path
        self.num_points=num_points
        self.crop_size=crop_size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        def getPoints(numPoints, prob_arr):
            prob_reshape = prob_arr.reshape(-1,)
            
            a = np.indices((prob_reshape.shape[0],)).T
            a = a.reshape((a.shape[0],))

            y_values = []
            x_values = []

            while len(x_values) < numPoints:
                val = np.random.choice(a, size = 1, p = prob_reshape)
                point = np.unravel_index(val, prob_arr.shape)

                y_threshold_amt = threshold * prob_arr.shape[0]
                if point[0][0] < y_threshold_amt or point[0][0] > prob_arr.shape[0] - y_threshold_amt:
                    continue
                
                x_threshold_amt = threshold * prob_arr.shape[1]
                if point[1][0] < x_threshold_amt or point[1][0] > prob_arr.shape[1] - x_threshold_amt:
                    continue

                x_values.append(point[1][0])
                y_values.append(point[0][0])
            
            return np.array((np.array(y_values), np.array(x_values)))

        def croppedImages(points, image):
            crops = []
            n = points.shape[1]
                
            for v in range(n):
                y1,x1 = points[:,v]

                # large crop image
                left = x1 - large_crop_dimension / 2
                top = y1 - large_crop_dimension / 2
                right = x1 + large_crop_dimension / 2
                bottom = y1 + large_crop_dimension / 2
                cropped_img = image.crop((left, top, right, bottom))

                crops.append(cropped_img)

            return crops

        
        path_vals = path.split('/')
        array_directory = 'salience_maps_16ids'
        prob_array = np.load('/checkpoint/mgahl/{}/{}/{}/{}.npy'.format(array_directory, path_vals[-3], path_vals[-2], path_vals[-1].split('.jpg')[0])

        points = getPoints(self.num_points, prob_arr)
        

        images = croppedImages(points, img)

        return (images)




    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
