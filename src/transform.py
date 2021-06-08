import albumentations
from albumentations.pytorch import ToTensor

def create_train_transform(flip,\
        noise,\
        cutout,\
        resize,\
        size = 224):
    translist=[]
    ### resize
    if flip:
        translist+=[albumentations.OneOf([
            albumentations.HorizontalFlip(),
            albumentations.RandomRotate90(),
            albumentations.VerticalFlip()],p=0.75)]

    ### noise
    if noise:
        translist+=[albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0,30.0))], p=0.75)]
        translist+=[albumentations.OneOf([
            albumentations.RandomBrightness(limit=0.2, p=0.65),
            albumentations.RandomContrast(limit=0.2, p=0.65),
            ])]
        translist+=[albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
            ], p=0.65)]
    if resize:
        translist+=[albumentations.Resize(size+30,size+30)] ## original image width : 300
        translist+=[albumentations.RandomCrop(size,size,always_apply=True)]
    ### cutout
    if cutout:
        translist+= [albumentations.Cutout(max_h_size = int(size * 0.2), max_w_size = int(size * 0.2), num_holes = 1,p=0.5)]

    ### normalized & totensor
    translist+=[albumentations.Normalize(mean = (0.5532, 0.5064, 0.4556), std = (0.2715, 0.2616, 0.2744))] # 아래에서 구한 값으로 
    translist+=[ToTensor()]
    transform = albumentations.Compose(translist)
    return transform

def create_validation_transform(resize,\
        size = 300):
    translist=[]
    ### resize
    if resize:
        translist+=[albumentations.Resize(size,size)]                                                                                    
    ### normalized
    translist+=[albumentations.Normalize(mean = (0.5532, 0.5064, 0.4556), std = (0.2715, 0.2616, 0.2744))] # 아래에서 구한 값으로 r
    translist+=[ToTensor()]
    transform = albumentations.Compose(translist)
    return transform


