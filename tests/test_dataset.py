from datasets.radar_dataset import RadarDataset


root = "/home/student/chipk/Radar"


for split in [
    "train",
    "val",
    "test",
]:

    dataset = RadarDataset(
        root,
        split=split
    )

    past, future = dataset[0]

    print()
    print(split)
    print(
        "size:",
        len(dataset)
    )

    print(
        "past:",
        past.shape,
        past.min().item(),
        past.max().item()
    )

    print(
        "future:",
        future.shape,
        future.min().item(),
        future.max().item()
    )