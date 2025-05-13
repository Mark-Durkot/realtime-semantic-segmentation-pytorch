import numpy as np
import albumentations as AT


def to_numpy(array):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    return array    


class Scale(AT.BasicTransform):
    def __init__(self, scale, interpolation=1, p=1, is_testing=False):
        super().__init__(p=p)
        self.scale = scale
        self.interpolation = interpolation
        self.is_testing = is_testing

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
        }

    def apply(self, img, **params):
        imgh, imgw, _ = img.shape
        new_imgh, new_imgw = int(imgh * self.scale), int(imgw * self.scale)
        aug = AT.Resize(height=new_imgh, width=new_imgw, interpolation=self.interpolation, p=1.0)
        return aug(image=img)['image']

    def apply_to_mask(self, mask, **params):
        imgh, imgw = mask.shape[:2]
        new_imgh, new_imgw = int(imgh * self.scale), int(imgw * self.scale)
        aug = AT.Resize(height=new_imgh, width=new_imgw, interpolation=self.interpolation, p=1.0)
        return aug(image=mask)['image']

    def get_transform_init_args_names(self):
        return ('scale', 'interpolation', 'is_testing')