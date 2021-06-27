import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import config

# number of subprocesses to use for data loading
num_workers = 0



def loadTestData(batch_size):
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_data = datasets.ImageFolder(config.test_dir, transform=test_transforms)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=num_workers)

    return test_loader
