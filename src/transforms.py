import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor

PATCH_SIZE = 256

def pre_transforms():
    result = [
        albu.Resize(height=PATCH_SIZE, width=PATCH_SIZE, always_apply=True),
    ]

    return result

def light_transforms():
    result = [
        albu.RandomBrightness(),
        albu.RandomContrast(),
        albu.HorizontalFlip(),

    ]
    return result

def spatial_transforms():
    result = [
        albu.ShiftScaleRotate(
            always_apply=False,
            p=0.5,
            shift_limit=(-0.059, 0.050),
            scale_limit=(-0.1, 0.07),
            rotate_limit=(-29, 29),
            interpolation=0,
            border_mode=1,
            value=(0, 0, 0),
            mask_value=None,
        ),
        albu.IAAPerspective(p=0.5),

    ]
    return result

def hard_transforms():
    result = [

        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf([
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
        ], p=0.4),
        albu.HorizontalFlip(),

        albu.ShiftScaleRotate(
            always_apply=False,
            p=0.5,
            shift_limit=(-0.059, 0.050),
            scale_limit=(-0.1, 0.07),
            rotate_limit=(-29, 29),
            interpolation=0,
            border_mode=1,
            value=(0, 0, 0),
            mask_value=None,
        ),

    ]
    return result

def post_transforms():
    return [albu.Resize(PATCH_SIZE, PATCH_SIZE, always_apply=True), albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result



def get_light_transforms():
    return compose([
        pre_transforms(),
        light_transforms(),
        post_transforms()
    ])
    
def get_spatial_transforms():
    return compose([
        pre_transforms(),
        spatial_transforms(),
        post_transforms()
    ])
    
def get_hard_transforms():
    return compose([
        pre_transforms(),
        hard_transforms(),
        post_transforms()
    ])


def get_valid_transforms():
    return compose([
        pre_transforms(),
        post_transforms()
    ])
