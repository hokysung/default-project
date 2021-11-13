import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_dataloaders(data_dir, imsize, batch_size, eval_size, num_workers=1):
    r"""
    Creates a dataloader from a directory containing image data.
    """

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    eval_dataset, train_dataset = torch.utils.data.random_split(
        dataset,
        [eval_size, len(dataset) - eval_size],
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_dataloader, eval_dataloader

def get_dataloaders_MNIST(data_dir, imsize, batch_size, eval_size, num_workers=1):
    # dataset = datasets.MNIST(root=data_dir, train=True, download=True,
    #     transform=transforms.ToTensor())

    # eval_dataset, train_dataset = torch.utils.data.random_split(
    #     dataset,
    #     [eval_size, len(dataset) - eval_size],
    # )

    # eval_dataloader = torch.utils.data.DataLoader(
    #     eval_dataset, batch_size=batch_size, num_workers=num_workers
    # )
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    # )

    # Image processing
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5),   # 3 for RGB channels
                                        std=(0.5))])

    # MNIST dataset
    mnist = datasets.MNIST(root=data_dir,
                                    train=True,
                                    transform=transform,
                                    download=True)

    eval_dataset, train_dataset = torch.utils.data.random_split(
        mnist,
        [eval_size, len(mnist) - eval_size],
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_dataloader, eval_dataloader