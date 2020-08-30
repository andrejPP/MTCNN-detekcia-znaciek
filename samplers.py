import torch
import numpy as np
import sys
import time
from torch.utils.data import Sampler


class BasicSampler(Sampler):
    """
        Simple sampler, create iter instance of random sequence of values from intervale specified
        by "start" and "end" arguments.
        Interval is closed so start and end values are included.

        Args:
            start: first value of interval
            end: last value of interval
            use: we can shrink interval into N random samples, N is specified
                by this argument
    """

    def __init__(self, start: int, end: int, use: int = None):
        self._val_arr = np.arange(start, end+1)
        if (use is not None) and (use > len(self._val_arr)):
            raise ValueError("\"use\" argument can't be larger than number of samples.")
        self._use = use

    def __iter__(self):
        if self._use is not None:
            np.random.shuffle(self._val_arr)
            return iter(self._val_arr[:self._use].tolist())

        random_idx = torch.randperm(len(self))
        return iter(self._val_arr[random_idx].tolist())

    def __len__(self):
        if self._use is not None:
            return self._use
        return len(self._val_arr)


class BatchSamplerMulti(Sampler):
    """
    Dataset contains three main types of samples. Negatives, positives, parts.
    Sampler should preserve chosen ration between those three types in every
    mini-batch and stop when the first category runs out of samples.
    Takes BasicSampler instance for each category with a given ratio. 

    Args:
        samplers: dict containing BasicSamplers instances as key and ratio as value
            example.   {sampler1_obj: 1/3,
                        sampler2_obj: 2/3}
        batch_size: size of one mini_batch
    """

    def __init__(self, samplers: dict, batch_size: int):
        self._samplers = samplers
        self._batch_size = batch_size
        self._size = 0

    def _initialize_iterators(self):
        """
        Setup sampler based on BasicSampler instances in a proper way, 
        so it will stop when the first one runs out of samples.
        For each sampler create an iterator and calculate how many samples should 
        be used in mini-batch and how many mini-batches it will last. 
        All this calculation is based on a given ratio. Store this informations in list.
        """

        samplers = []
        iterators = []
        samples_per_batch = []
        num_of_batches = []
        for sampler, ratio in self._samplers.items():
            samplers.append(sampler)
            iterators.append(iter(sampler))
            samples_per_batch.append(int(self._batch_size * ratio))
            num_of_batches.append(int(len(sampler) / samples_per_batch[-1]))

        # Find the index of sampler which will last the smallest number of batches.
        idx = np.argmin(num_of_batches)
        self._size = num_of_batches[idx]

        # Fix the number of samples per batch in case it is off by same value (due to rounding error) 
        # compared to intended size of batch. 
        diff = self._batch_size - sum(samples_per_batch)
        idx = np.argmax(num_of_batches)
        samples_per_batch[idx] += diff

        # Calculate how many mini-batches it will last with the new value
        num = len(samplers[idx]) // samples_per_batch[idx]

        if num < self._size:
            raise ValueError("Number of mini-batches for {} sampler is less then size of mini-batches".format(idx))

        return iterators, samples_per_batch

    def __iter__(self):
        """
        In every iteration generate new list of numbers. Each number
        corespond to index of image in dataset.
        """

        batch = []
        iterators, samples_per_batch = self._initialize_iterators()
        for _ in range(len(self)):
            for iterator_idx in range(len(iterators)):

                counter = samples_per_batch[iterator_idx]

                for sample_idx in iterators[iterator_idx]:
                    batch.append(sample_idx)
                    counter -= 1
                    if counter == 0:
                        break

            np.random.shuffle(batch)
            yield batch
            batch = []

    def __len__(self):
        self._initialize_iterators()
        return self._size

# e.g. simple usage of classes above
# if __name__ == "__main__":
#
#     test_input = {'negatives': [0, 5625], 'positives': [5626, 6477], 'z_parts': [6478, 7329]}
#     print(test_input)
#     neg = test_input['negatives']
#     pos = test_input['positives']
#     part = test_input['z_parts']
#
#     neg_sampler = BasicSampler(start=neg[0], end=neg[1], use=3300)
#     pos_sampler = BasicSampler(start=pos[0], end=pos[1])
#     part_sampler = BasicSampler(start=part[0], end=part[1])
#     # print(len(neg_sampler))
#     # print(len(pos_sampler))
#     # print(len(part_sampler))
#     samplers = {neg_sampler: 0.6,
#                 pos_sampler: 0.2,
#                 part_sampler: 0.2}
#     batch_sampler = BatchSamplerMulti(samplers=samplers, batch_size=16)
#
#     # print(len(batch_sampler))
#     # for index,i in enumerate(batch_sampler):
#     #     # continue
#     #     print(index,i)
#     for index,i in enumerate(batch_sampler):
#         # continue
#         print(index,i)