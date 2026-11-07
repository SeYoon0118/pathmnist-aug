
from torchvision import transforms
from medmnist import PathMNIST

def get_transform(experiment="A"):
    # A: 원본
    if experiment == "A":
        return transforms.ToTensor()

    # B, C 공통: 방향 증강
    elif experiment in ["B", "C"]:
        operations = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ])
        ]

        # C: 방향 증강 + 색상 증강
        if experiment == "C":
            operations.append(
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.01
                    )
                ], p=0.8)
            )

        operations.append(transforms.ToTensor())
        return transforms.Compose(operations)

    else:
        raise ValueError("A, B, C 중 선택하세요.")


def get_datasets(experiment="A"):
    train = PathMNIST(
        split="train",
        download=True,
        transform=get_transform(experiment)
    )

    # 검증·테스트에는 증강 적용 안 함
    val = PathMNIST(
        split="val",
        download=True,
        transform=get_transform("A")
    )

    test = PathMNIST(
        split="test",
        download=True,
        transform=get_transform("A")
    )

    return train, val, test
