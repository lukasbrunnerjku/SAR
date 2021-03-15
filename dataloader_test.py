import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(100)

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            print('worker_id {} calling with index {}'.format(worker_id, index))
            if worker_id == 0:
                print('slowing down worker0')
                a = 0.
                for idx in range(10000000):
                    a += idx
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    dataset = MyDataset()
    loader = DataLoader(dataset, batch_size=4, num_workers=4)

    for data in loader:
        print(data)

    # in the console log below we can see that even if a worker is slowed down the other 
    # workers won't yield anything even if they have processed enough data to send, instead
    # they continue with work but put the results of their work into a queue!
    # data is thus received in order! which is a must have for our drone flight example!
    """
    worker_id 0 calling with index 0
    slowing down worker0
    worker_id 2 calling with index 8
    worker_id 2 calling with index 9
    worker_id 2 calling with index 10
    worker_id 2 calling with index 11
    worker_id 1 calling with index 4
    worker_id 1 calling with index 5
    worker_id 1 calling with index 6
    worker_id 1 calling with index 7
    worker_id 2 calling with index 24
    worker_id 2 calling with index 25
    worker_id 1 calling with index 20
    worker_id 1 calling with index 21
    worker_id 2 calling with index 26
    worker_id 2 calling with index 27
    worker_id 1 calling with index 22
    worker_id 1 calling with index 23
    worker_id 3 calling with index 12
    worker_id 3 calling with index 13
    worker_id 3 calling with index 14
    worker_id 3 calling with index 15
    worker_id 3 calling with index 28
    worker_id 3 calling with index 29
    worker_id 3 calling with index 30
    worker_id 3 calling with index 31
    worker_id 0 calling with index 1
    slowing down worker0
    worker_id 0 calling with index 2
    slowing down worker0
    worker_id 0 calling with index 3
    slowing down worker0
    worker_id 0 calling with index 16
    slowing down worker0
    tensor([0, 1, 2, 3])
    tensor([4, 5, 6, 7])
    worker_id 1 calling with index 36
    tensor([ 8,  9, 10, 11])
    worker_id 2 calling with index 40
    worker_id 3 calling with index 44
    tensor([12, 13, 14, 15])
    worker_id 2 calling with index 41
    worker_id 1 calling with index 37
    ...
    ...
    """