from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop
import torchvision.transforms.v2  as transforms
import torch

def get_aug():
    train_sz = 1024
    iou_crop = CustomRandomIoUCrop(min_scale=0.3, 
                               max_scale=1.0, 
                               min_aspect_ratio=0.5, 
                               max_aspect_ratio=2.0, 
                               sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                               trials=400, 
                               jitter_factor=0.25)

# Create a `ResizeMax` object
    resize_max = ResizeMax(max_sz=train_sz)

    # Create a `PadSquare` object
    pad_square = PadSquare(shift=True, fill=0)

        
        # Compose transforms for data augmentation
    data_aug_tfms = transforms.Compose([
        iou_crop,
        transforms.ColorJitter(
                brightness = (0.875, 1.125),
                contrast = (0.5, 1.5),
                saturation = (0.5, 1.5),
                hue = (-0.05, 0.05),
        ),
        transforms.RandomGrayscale(),
        transforms.RandomEqualize(),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 360)),
        ],
    )

    # Compose transforms to resize and pad input images
    resize_pad_tfm = transforms.Compose([
        resize_max, 
        pad_square,
        transforms.Resize([train_sz] * 2, antialias=True)
    ])

    # Compose transforms to sanitize bounding boxes and normalize input data
    final_tfms = transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.SanitizeBoundingBoxes(), #remove all the bboxes not in the image or small 
    ])

    # Define the transformations for training and validation datasets
    train_tfms = transforms.Compose([
        data_aug_tfms, 
        resize_pad_tfm, 
        final_tfms
    ])
    valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms])
    
    return train_tfms, valid_tfms