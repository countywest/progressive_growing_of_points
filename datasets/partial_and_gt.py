# code from PCN, TopNet
# revised by Hyeontae Son
import numpy as np
import tensorflow as tf
from tensorpack import dataflow
import os
import h5py
import torch
from open3d import *

### For PCN dataset ###
def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


class PreprocessData(dataflow.ProxyDataFlow):
    def __init__(self, ds, input_size, output_size):
        super(PreprocessData, self).__init__(ds)
        self.input_size = input_size
        self.output_size = output_size

    def get_data(self):
        for id, input, gt in self.ds.get_data():
            input = resample_pcd(input, self.input_size)
            gt = resample_pcd(gt, self.output_size)
            yield id, input, gt


class BatchData(dataflow.ProxyDataFlow):
    def __init__(self, ds, batch_size, input_size, gt_size, remainder=False, use_list=False):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size = input_size
        self.gt_size = gt_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)

    def _aggregate_batch(self, data_holder, use_list=False):
        ''' Concatenate input points along the 0-th dimension
            Stack all other data along the 0-th dimension
        '''
        ids = np.stack([x[0] for x in data_holder])
        inputs = np.stack([resample_pcd(x[1], self.input_size) for x in data_holder]).astype(np.float32)
        gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(np.float32)
        return ids, inputs, gts


def lmdb_dataflow(lmdb_path, batch_size, input_size, output_size, is_training, test_speed=False):
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)    
    size = df.size()
    if is_training:
        df = dataflow.LocallyShuffleData(df, buffer_size=2000)
        df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
    df = BatchData(df, batch_size, input_size, output_size)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, nr_proc=8)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size


def get_queued_data(generator, dtypes, shapes, queue_capacity=10):
    assert len(dtypes) == len(shapes), 'dtypes and shapes must have the same length'
    queue = tf.FIFOQueue(queue_capacity, dtypes, shapes)
    placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)]
    enqueue_op = queue.enqueue(placeholders)
    close_op = queue.close(cancel_pending_enqueues=True)
    feed_fn = lambda: {placeholder: value for placeholder, value in zip(placeholders, next(generator))}
    queue_runner = tf.contrib.training.FeedingQueueRunner(queue, [enqueue_op], close_op, feed_fns=[feed_fn])
    tf.train.add_queue_runner(queue_runner)
    return queue.dequeue()

### For PCN Test Dataset ###
class PCNTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_data_path, num_input_points):
        with open(os.path.join(root_data_path, 'test.list')) as f:
            self.id_list_path = f.read().splitlines()
        self.data_dir = os.path.join(root_data_path, 'test')
        self.partial_list = [os.path.join(self.data_dir, 'partial', f + ".pcd") for f in self.id_list_path]
        self.gt_list = [os.path.join(self.data_dir, 'complete', f + ".pcd") for f in self.id_list_path]
        self.num_input_points = num_input_points

    def __len__(self):
        return len(self.partial_list)

    def __getitem__(self, index):
        partial_filename = self.partial_list[index]
        partial_pcd = io.read_point_cloud(partial_filename)
        gt_filename = self.gt_list[index]
        gt_pcd = io.read_point_cloud(gt_filename)
        parsed_list = partial_filename.split('/')
        synset_id = parsed_list[-2]
        obj_id = parsed_list[-1].split('.')[0]
        return {'id': synset_id + '_' + obj_id,
                'partial': resample_pcd(np.array(partial_pcd.points), self.num_input_points).astype(np.float32),
                'gt': np.array(gt_pcd.points).astype(np.float32)
                }

### For TopNet dataset ###
class TopNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_data_path, type):
        with open(os.path.join(root_data_path, type + '.list')) as f:
            self.id_list_path = f.read().splitlines()
        self.data_dir = os.path.join(root_data_path, type)
        self.partial_list = [os.path.join(self.data_dir, 'partial', f + ".h5") for f in self.id_list_path]
        self.gt_list = [os.path.join(self.data_dir, 'gt', f + ".h5") for f in self.id_list_path]

    def __len__(self):
        return len(self.partial_list)

    def __getitem__(self, index):
        partial_filename = self.partial_list[index]
        gt_filename = self.gt_list[index]
        parsed_list = partial_filename.split('/')
        synset_id = parsed_list[-2]
        obj_id = parsed_list[-1].split('.')[0]
        return {'id':synset_id + '_' + obj_id,
                'partial': TopNetDataset.h5_reader(partial_filename),
                'gt': TopNetDataset.h5_reader(gt_filename)
                }

    @staticmethod
    def h5_reader(path):
        with h5py.File(path, 'r') as f:
            data = f.get('data').value.astype(np.float32)
            return data