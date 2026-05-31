import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

FABRIC_LABELS = {
    0: "N/A",
    1: "denim",
    2: "cotton",
    3: "leather",
    4: "furry",
    5: "knitted",
    6: "chiffon",
    7: "other"
}

PATTERN_LABELS = {
    0: "N/A",
    1: "solid",
    2: "floral",
    3: "plaid",
    4: "stripe",
    5: "print",
    6: "graphic",
    7: "other"
}

SHAPE_LABELS = {
    0: "N/A",
    1: "sleeveless",
    2: "short_sleeve",
    3: "medium_sleeve",
    4: "long_sleeve",
    5: "long_pants",
    6: "short_pants",
    7: "skirt",
    8: "dress"
}

def get_transforms(image_size=512, split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

def get_mask_transforms(image_size=512):
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=Image.NEAREST
        ),
        transforms.ToTensor(),
    ])

class FashionDataset(Dataset):

    def __init__(
        self,
        curated_path,
        split="train",
        val_ratio=0.1,
        image_size=512,
        seed=42
    ):

        self.curated_path = curated_path
        self.image_size = image_size
        self.split = split

        all_samples = sorted([
            d for d in os.listdir(curated_path)
            if os.path.isdir(os.path.join(curated_path, d))
        ])

        random.seed(seed)
        random.shuffle(all_samples)

        val_count = int(len(all_samples) * val_ratio)

        val_samples = all_samples[:val_count]
        train_samples = all_samples[val_count:]

        self.samples = (
            train_samples
            if split == "train"
            else val_samples
        )

        self.img_transform = get_transforms(
            image_size,
            split
        )

        self.mask_transform = get_mask_transforms(
            image_size
        )

        print(
            f"[FashionDataset] {split}: {len(self.samples)} samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample_id = self.samples[idx]

        folder = os.path.join(
            self.curated_path,
            sample_id
        )

        img_path = os.path.join(
            folder,
            "image.jpg"
        )

        if not os.path.exists(img_path):
            img_path = os.path.join(
                folder,
                "image.png"
            )

        image = Image.open(
            img_path
        ).convert("RGB")

        image = self.img_transform(image)

        densepose = Image.open(
            os.path.join(
                folder,
                "densepose.png"
            )
        ).convert("RGB")

        densepose = self.mask_transform(
            densepose
        )

        segm = Image.open(
            os.path.join(
                folder,
                "segm.png"
            )
        ).convert("RGB")

        segm = self.mask_transform(
            segm
        )

        with open(
            os.path.join(folder, "caption.txt"),
            "r",
            encoding="utf-8"
        ) as f:

            caption = f.read().strip()

        with open(
            os.path.join(folder, "shape.txt"),
            "r"
        ) as f:

            shape = torch.tensor(
                [
                    int(v)
                    for v in f.read().strip().split()
                ],
                dtype=torch.long
            )

        with open(
            os.path.join(folder, "fabric.txt"),
            "r"
        ) as f:

            fabric = torch.tensor(
                [
                    int(v)
                    for v in f.read().strip().split()
                ],
                dtype=torch.long
            )

        with open(
            os.path.join(folder, "pattern.txt"),
            "r"
        ) as f:

            pattern = torch.tensor(
                [
                    int(v)
                    for v in f.read().strip().split()
                ],
                dtype=torch.long
            )

        with open(
            os.path.join(folder, "keypoints_loc.txt"),
            "r"
        ) as f:

            kp_vals = [
                int(v)
                for v in f.read().strip().split()
            ]

            kp_loc = torch.tensor(
                [
                    [kp_vals[i], kp_vals[i + 1]]
                    for i in range(
                        0,
                        len(kp_vals) - 1,
                        2
                    )
                ],
                dtype=torch.float32
            )

        with open(
            os.path.join(folder, "keypoints_vis.txt"),
            "r"
        ) as f:

            kp_vis = torch.tensor(
                [
                    int(v)
                    for v in f.read().strip().split()
                ],
                dtype=torch.float32
            )

        return {
            "image": image,
            "densepose": densepose,
            "segm": segm,
            "caption": caption,
            "shape": shape,
            "fabric": fabric,
            "pattern": pattern,
            "keypoints_loc": kp_loc,
            "keypoints_vis": kp_vis,
            "sample_id": sample_id
        }

def get_dataloader(
    curated_path,
    batch_size=8,
    image_size=512,
    num_workers=4,
    val_ratio=0.1
):

    train_ds = FashionDataset(
        curated_path,
        split="train",
        val_ratio=val_ratio,
        image_size=image_size
    )

    val_ds = FashionDataset(
        curated_path,
        split="val",
        val_ratio=val_ratio,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\nDataLoader ready:")
    print(
        f"Train batches: {len(train_loader)} ({len(train_ds)} samples)"
    )
    print(
        f"Val batches: {len(val_loader)} ({len(val_ds)} samples)"
    )

    return train_loader, val_loader

if __name__ == "__main__":

    CURATED = "./curated_dataset"

    train_loader, val_loader = get_dataloader(
        CURATED,
        batch_size=4,
        num_workers=0
    )

    print("\nTesting one batch...")

    batch = next(iter(train_loader))

    print(
        f"image shape: {batch['image'].shape}"
    )

    print(
        f"densepose shape: {batch['densepose'].shape}"
    )

    print(
        f"segm shape: {batch['segm'].shape}"
    )

    print(
        f"shape labels: {batch['shape'].shape}"
    )

    print(
        f"fabric labels: {batch['fabric'].shape}"
    )

    print(
        f"pattern labels: {batch['pattern'].shape}"
    )

    print(
        f"keypoints_loc shape: {batch['keypoints_loc'].shape}"
    )

    print(
        f"keypoints_vis shape: {batch['keypoints_vis'].shape}"
    )

    print(
        f"sample captions[0]: {batch['caption'][0]}"
    )

    print("\nPipeline test PASSED.")