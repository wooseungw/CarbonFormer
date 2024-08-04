import torch.distributed as dist
import torch
from data.NIA import NIADataset
from data.carbon_dataset import CarbonDataset


def build_data_loader(dataset, base_dir, image_size, split, batch_size, num_workers, local_rank, cfg, shuffle=True):


    if dataset == 'NIA':
        dataset = NIADataset(base_dir, image_size, split)
    elif dataset =='carbon':
        dataset = CarbonDataset(base_dir+split, cfg)

    print(f"local rank {local_rank} / global rank {dist.get_rank()} successfully build %s dataset" % (split))

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return data_loader