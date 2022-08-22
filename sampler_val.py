import math
import torch
from torch.utils.data.sampler import RandomSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        # self.smallest_dataset_size = min([len(cur_dataset.samples) for cur_dataset in dataset.datasets])
        self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        datasets_size = []
        
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            datasets_size.append(len(cur_dataset))
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        # interactions = math.ceil(self.smallest_dataset_size / self.batch_size)
        interactions = math.ceil(self.largest_dataset_size / self.batch_size)
        
        # epoch_samples = self.smallest_dataset_size * self.number_of_datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            
            for i in range(self.number_of_datasets):
                
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                samples_to_grab =  math.ceil(datasets_size[i] / interactions)

                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        break
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
    
    
    ### !!!! Retrive diferente number of samples for each dataset but have the same number of interactions ########