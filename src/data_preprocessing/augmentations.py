import albumentations as A

def get_color_augmentations():
    """Аугментации только для цвета изображения"""
    return A.Compose([
        A.OneOf([
            A.GaussNoise(),
            A.MultiplicativeNoise(),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.5),
        A.HueSaturationValue(p=0.4),
        A.RandomGamma(p=0.3),
    ])

def get_shape_augmentations():
    """Аугментации для формы изображения"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.25),
            A.MedianBlur(blur_limit=3, p=0.25),
            A.Blur(blur_limit=3, p=0.25),
        ], p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=25,
            p=0.75
        ),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
        ], p=0.4),
    ], additional_targets={'image': 'image', 'annotation': 'image'})
