# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection
# Orr Zohar, Jackson Wang, Serena Yeung
# ------------------------------------------------------------------------

from datasets.coco import make_coco_transforms, make_owl_vit_transforms
from .torchvision_datasets.real_world import RWDetection
from .torchvision_datasets.open_world import OWDetection


def build_dataset(args, image_set):
    return OWDetection(args,
                       args.data_root,
                       image_set=image_set,
                       transforms=make_owl_vit_transforms(args.image_resize),
                       dataset=args.dataset)
    '''
    return RWDetection(args,
                       args.data_root,
                       image_set=image_set,
                       transforms=make_owl_vit_transforms(args.image_resize),
                       dataset=args.dataset)
    '''
