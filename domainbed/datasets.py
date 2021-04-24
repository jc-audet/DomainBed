# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import copy as cp


# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset
# from wilds.datasets.iwildcam_dataset import IWildCamDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    "CFMNIST",
    "ACMNIST",
    "CSMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # "CelebA",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    "WILDSIWildCams",
    "WILDSCelebA",
    #Other
    "Spirals",
    "ChainEquationModel"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 10     # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['-90%', '+90%', '+80%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.9, 0.1, 0.2],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class CFMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0.9', '0.2','0.1']

    def __init__(self, root, test_envs, hparams):
        super(CFMNIST, self).__init__(root, [0.9, 0.2, 0.1],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.p_label = 0.25

    def color_dataset(self, images, labels, environment):
        # Convert y>5 to 1 and y<5 to 0.
        self.p_label = 0.25
        y = (labels >= 5).float()

        num_samples = len(y)
        h = np.random.binomial(1, self.p_label, (num_samples, 1))
        h1 = np.random.binomial(1, environment, (num_samples, 1))
        y_mod = np.abs(y.unsqueeze(1) - h)
        z = np.logical_xor(h1, h)

        red = np.where(z == 1)[0]
        tsh = 0.0
        chR = cp.deepcopy(images[red, :])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(images[red, :])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(images[red, :])
        chB[chB > tsh] = 0
        r = np.concatenate((chR.unsqueeze(3), chG.unsqueeze(3)), axis=3)

        green = np.where(z == 0)[0]
        tsh = 0.0
        chR1 = cp.deepcopy(images[green, :])
        chR1[chR1 > tsh] = 0
        chG1 = cp.deepcopy(images[green, :])
        chG1[chG1 > tsh] = 1
        chB1 = cp.deepcopy(images[green, :])
        chB1[chB1 > tsh] = 0
        g = np.concatenate((chR1.unsqueeze(3), chG1.unsqueeze(3)), axis=3)

        dataset = np.transpose(np.concatenate((r, g), axis=0), (0, 3, 1, 2))
        dataset = torch.tensor(dataset, dtype=torch.float32)

        labels = torch.squeeze(torch.tensor(np.concatenate((y_mod[red], y_mod[green]), axis=0), dtype=torch.long), 1)

        return TensorDataset(dataset,labels)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class ACMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0.9', '0.2','0.1']

    def __init__(self, root, test_envs, hparams):
        super(ACMNIST, self).__init__(root, [0.9, 0.2, 0.1],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.p_label = 0.25


    def color_dataset(self, images, labels, environment):
        # Convert y>5 to 1 and y<5 to 0.
        self.p_label = 0.25
        y = (labels >= 5).float()
        num_samples = len(y)

        y_mod = np.abs(y.unsqueeze(1) - np.random.binomial(1, self.p_label, (num_samples, 1)))
        z = np.abs(y_mod - np.random.binomial(1, environment, (num_samples, 1)))
        red = np.where(z == 1)[0]

        tsh = 0.0
        chR = cp.deepcopy(images[red, :])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(images[red, :])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(images[red, :])
        chB[chB > tsh] = 0
        r = np.concatenate((chR.unsqueeze(3), chG.unsqueeze(3)), axis=3)

        green = np.where(z == 0)[0]
        tsh = 0.0
        chR1 = cp.deepcopy(images[green, :])
        chR1[chR1 > tsh] = 0
        chG1 = cp.deepcopy(images[green, :])
        chG1[chG1 > tsh] = 1
        chB1 = cp.deepcopy(images[green, :])
        chB1[chB1 > tsh] = 0
        g = np.concatenate((chR1.unsqueeze(3), chG1.unsqueeze(3)), axis=3)

        dataset = np.transpose(np.concatenate((r, g), axis=0), (0, 3, 1, 2))
        dataset = torch.tensor(dataset, dtype=torch.float32)

        labels = torch.squeeze(torch.tensor(np.concatenate((y_mod[red], y_mod[green]), axis=0), dtype=torch.long),1)

        return TensorDataset(dataset,labels)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class CSMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0.9', '0.2','0.1']

    def __init__(self, root, test_envs, hparams):
        super(CSMNIST, self).__init__(root, [0.9, 0.2, 0.1],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (3, 28, 28,)
        self.num_classes = 2
        self.p_label = 0.25


    def color_dataset(self, images, labels, environment):
        # prob_label we retain from other classes for simplicity but is not relevant for this class
        self.p_label = 0.25
        # Convert y>5 to 1 and y<5 to 0.
        y = (labels >= 5).float()

        num_samples = len(y)
        z_color = np.random.binomial(1, 0.5, (num_samples, 1))  # sample color for each sample
        w_comb = 1 - np.logical_xor(y.unsqueeze(1), z_color)  # compute xor of label and color and negate it

        selection_0 = np.where(w_comb == 0)[0]  # indices where -xor is zero
        selection_1 = np.where(w_comb == 1)[0]  # indices were -xor is one
        ns0 = np.shape(selection_0)[0]
        ns1 = np.shape(selection_1)[0]
        final_selection_0 = selection_0[np.where(np.random.binomial(1, environment, (ns0, 1)) == 1)[
            0]]  # -xor =0 then select that point with probability prob_e
        final_selection_1 = selection_1[np.where(np.random.binomial(1, 1 - environment, (ns1, 1)) == 1)[
            0]]  # -xor =0 then select that point with probability 1-prob_e

        final_selection = np.concatenate((final_selection_0, final_selection_1),
                                         axis=0)  # indices of the final set of points selected
        z_color_final = z_color[final_selection]  # colors of the final set of selected points
        y = y[final_selection]  # labels of the final set of selected points
        images = images[final_selection]  # gray scale image of the final set of selected points

        ### color the points x based on z_color_final
        red = np.where(z_color_final == 0)[0]  # select the points with z_color_final=0 to set them to red color
        green = np.where(z_color_final == 1)[0]  # select the points with z_color_final=1 to set them to green color

        num_samples_final = np.shape(y)[0]

        tsh = 0.5
        chR = cp.deepcopy(images[red, :])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(images[red, :])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(images[red, :])
        chB[chB > tsh] = 0
        r = np.concatenate((chR.unsqueeze(3), chG.unsqueeze(3), chB.unsqueeze(3)), axis=3)

        tsh = 0.5
        chR1 = cp.deepcopy(images[green, :])
        chR1[chR1 > tsh] = 0
        chG1 = cp.deepcopy(images[green, :])
        chG1[chG1 > tsh] = 1
        chB1 = cp.deepcopy(images[green, :])
        chB1[chB1 > tsh] = 0
        g = np.concatenate((chR1.unsqueeze(3), chG1.unsqueeze(3), chB1.unsqueeze(3)), axis=3)

        dataset = np.transpose(np.concatenate((r, g), axis=0), (0,3,1,2))
        dataset = torch.tensor(dataset, dtype=torch.float32)
        labels = torch.tensor(np.concatenate((y[red], y[green]), axis=0), dtype=torch.long)
        return TensorDataset(dataset,labels)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class Spirals(MultipleDomainDataset):
    CHECKPOINT_FREQ = 10
    ENVIRONMENTS = [str(i) for i in range(8)]

    def __init__(self, root, test_env, hparams):
        super().__init__()
        self.datasets = []

        test_dataset = self.make_tensor_dataset(env='test')
        self.datasets.append(test_dataset)
        for env in self.ENVIRONMENTS:
            env_dataset = self.make_tensor_dataset(env=env)
            self.datasets.append(env_dataset)
        
        self.input_shape = (10,)
        self.num_classes = 2

    def make_tensor_dataset(self, env, n_examples=1024, n_envs=16, n_revolutions=3, n_dims=8,
                        flip_first_signature=False,
                        seed=0):

        if env == 'test':
            inputs, labels = self.generate_environment(2000,
                                            n_rotations=n_revolutions,
                                            env=env,
                                            n_envs=n_envs,
                                            n_dims_signatures=n_dims,
                                            seed=2 ** 32 - 1
                                            )
        else:
            inputs, labels = self.generate_environment(n_examples,
                                            n_rotations=n_revolutions,
                                            env=env,
                                            n_envs=n_envs,
                                            n_dims_signatures=n_dims,
                                            seed=seed
                                            )
        if flip_first_signature:
            inputs[:1, 2:] = -inputs[:1, 2:]

        return TensorDataset(torch.tensor(inputs), torch.tensor(labels))

    def generate_environment(self, n_examples, n_rotations, env, n_envs,
                        n_dims_signatures,
                        seed=None):
        """
        env must either be "test" or an int between 0 and n_envs-1
        n_dims_signatures: how many dimensions for the signatures (spirals are always 2)
        seed: seed for numpy
        """
        assert env == 'test' or 0 <= int(env) < n_envs

        # Generate fixed dictionary of signatures
        rng = np.random.RandomState(seed)

        signatures_matrix = rng.randn(n_envs, n_dims_signatures)

        radii = rng.uniform(0.08, 1, n_examples)
        angles = 2 * n_rotations * np.pi * radii

        labels = rng.randint(0, 2, n_examples)
        angles = angles + np.pi * labels

        radii += rng.uniform(-0.02, 0.02, n_examples)
        xs = np.cos(angles) * radii
        ys = np.sin(angles) * radii

        if env == 'test':
            signatures = rng.randn(n_examples, n_dims_signatures)
        else:
            env = int(env)
            signatures_labels = np.array(labels * 2 - 1).reshape(1, -1)
            signatures = signatures_matrix[env] * signatures_labels.T

        signatures = np.stack(signatures)
        mechanisms = np.stack((xs, ys), axis=1)
        mechanisms /= mechanisms.std(axis=0)  # make approx unit variance (signatures already are)
        inputs = np.hstack((mechanisms, signatures))

        return inputs.astype(np.float32), labels.astype(np.long)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 5
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 5
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 50
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 5
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 5
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


class WILDSIWildCam(WILDSDataset):
    ENVIRONMENTS = [ str(i) for i in range(323)]
    def __init__(self, root, test_envs, hparams):
        dataset = IWildCamDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


class WILDSCelebA(WILDSDataset):
    ENVIRONMENTS = [ 'blond', 'not blond']
    def __init__(self, root, test_envs, hparams):
        dataset = IWildCamDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

            

class ChainEquationModel(MultipleDomainDataset):
    """
    Example 1 from the original IRM paper (https://arxiv.org/pdf/1907.02893.pdf)
    Implementation taken from the paper code release (https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/experiment_synthetic/sem.py)
    """
    N_STEPS = 50000
    CHECKPOINT_FREQ = 10
    ENVIRONMENTS = ['0.2', '2.', '5.']

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.hetero = hparams['hetero']
        self.hidden = hparams['hidden']
        self.dim = hparams['dim'] // 2
        self.num_classes = 1
        self.input_shape = (hparams['dim'],)

        if hparams['ones']:
            self.wxy = torch.eye(self.dim)
            self.wyz = torch.eye(self.dim)
        else:
            self.wxy = torch.randn(self.dim, self.dim) / hparams['dim']
            self.wyz = torch.randn(self.dim, self.dim) / hparams['dim']

        if hparams['scramble']:
            self.scramble, _ = torch.qr(torch.randn(hparams['dim'], hparams['dim']))
        else:
            self.scramble = torch.eye(hparams['dim'])

        if hparams['hidden']:
            self.whx = torch.randn(self.dim, self.dim) / hparams['dim']
            self.why = torch.randn(self.dim, self.dim) / hparams['dim']
            self.whz = torch.randn(self.dim, self.dim) / hparams['dim']
        else:
            self.whx = torch.eye(self.dim, self.dim)
            self.why = torch.zeros(self.dim, self.dim)
            self.whz = torch.zeros(self.dim, self.dim)

        ## Create dataset
        self.datasets = []

        for e in self.ENVIRONMENTS:
            self.datasets.append(self(float(e)))
        

    def solution(self):
        w = torch.cat((self.wxy.sum(1), torch.zeros(self.dim))).view(-1, 1)
        return w, self.scramble

    def __call__(self, env, n = 1000):
        h = torch.randn(n, self.dim) * env
        x = h @ self.whx + torch.randn(n, self.dim) * env

        if self.hetero:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim) * env
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim)
        else:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim)
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim) * env

        return TensorDataset(torch.cat((x, z), 1) @ self.scramble, y.sum(1, keepdim=True))

    
