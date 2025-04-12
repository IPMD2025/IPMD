from torchvision import transforms as T
import random, math


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def get_transform(args):
    transform_train = T.Compose([
        T.Resize((args.height, args.width)),
        # T.RandomHorizontalFlip(p=args.horizontal_flip_pro),
        T.Pad(padding=args.pad_size),
        T.RandomCrop((args.height, args.width)),

        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=args.random_erasing_pro, mean=[0.0, 0.0, 0.0])
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_shape = T.Compose([
        T.Resize((args.height, args.width),antialias=True),
        # T.RandomHorizontalFlip(p=args.horizontal_flip_pro),
        T.Pad(padding=args.pad_size),
        T.RandomCrop((args.height, args.width))])
    
    transform_value = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=args.random_erasing_pro, mean=[0.0, 0.0, 0.0])
    ])

    return transform_train, transform_test,transform_shape,transform_value

# def get_transform(args):
#     """Build transforms

#     Args:
#     - height (int): target image height.
#     - width (int): target image width.
#     - is_train (bool): train or test phase.
#     """

#     # use imagenet mean and std as default
#     imagenet_mean = [0.485, 0.456, 0.406]
#     imagenet_std = [0.229, 0.224, 0.225]
#     # imagenet_mean = [0.5, 0.5, 0.5]
#     # imagenet_std = [0.5, 0.5, 0.5]
#     # normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

#     transform_train = T.Compose([
#         # transforms += [Random2DTranslation(height, width)]
#         # transforms += [RandomHorizontalFlip()]
#         T.Resize((args.height, args.width)),
#         T.ToTensor(),
#         T.Normalize(mean=imagenet_mean, std=imagenet_std)
#         ])
#         # transforms += [RandomErasing()]  # RGB has this aug,,,,,,Not useful for GaitReID....
#     transform_test = T.Compose([
#         T.Resize((args.height, args.width)),
#         T.ToTensor(),
#         T.Normalize(mean=imagenet_mean, std=imagenet_std),
#     ])

#     # transforms = Compose(transforms)

#     return transform_train, transform_test
