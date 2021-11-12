import torchvision

dataset = torchvision.datasets.MNIST(root="../", train=True, download=True)
breakpoint()