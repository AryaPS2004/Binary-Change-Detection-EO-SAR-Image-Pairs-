import albumentations as A


def get_train_transforms(image_size=256):

    transforms = A.Compose(

        [
            A.Resize(image_size, image_size),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.ElasticTransform(
                alpha=1,
                sigma=50,
                p=0.2
            ),

        ],

        additional_targets={
            "sar": "image"
        }

    )

    return transforms


def get_val_transforms(image_size=256):

    transforms = A.Compose(

        [
            A.Resize(image_size, image_size),
        ],

        additional_targets={
            "sar": "image"
        }

    )

    return transforms