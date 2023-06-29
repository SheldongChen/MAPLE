# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np
from PIL import Image, ImageOps
import threading

import queue
from torch.utils.data import DataLoader

import random
import torch.nn.functional as F

"""
#based on http://stackoverflow.com/questions/7323664/python-generator-pre-fetch
This is a single-function package that transforms arbitrary generator into a background-thead generator that 
prefetches several batches of data in a parallel background thead.

This is useful if you have a computationally heavy process (CPU or GPU) that 
iteratively processes minibatches from the generator while the generator 
consumes some other resource (disk IO / loading from database / more CPU if you have unused cores). 

By default these two processes will constantly wait for one another to finish. If you make generator work in 
prefetch mode (see examples below), they will work in parallel, potentially saving you your GPU time.
We personally use the prefetch generator when iterating minibatches of data for deep learning with PyTorch etc.

Quick usage example (ipython notebook) - https://github.com/justheuristic/prefetch_generator/blob/master/example.ipynb
This package contains this object
 - BackgroundGenerator(any_other_generator[,max_prefetch = something])
"""


class BackgroundGenerator(threading.Thread):
    """
    the usage is below
    >> for batch in BackgroundGenerator(my_minibatch_iterator):
    >>    doit()
    More details are written in the BackgroundGenerator doc
    >> help(BackgroundGenerator)
    """

    def __init__(self, generator, local_rank, max_prefetch=30):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may raise GIL and zero-out the
        benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it
        outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving
        URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep
        stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until
        one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work
        slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size
        unless dequeued quickly enough.
        """
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(
            local_rank
        )  # create a new cuda stream in each process
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            # avoid re-entrance or ill-conditioned thread state
            return

        # Set exit event to True for background threading stopping
        self.iter.exit_event.set()

        # Exhaust all remaining elements, so that the queue becomes empty,
        # and the thread should quit
        for _ in self.iter:
            pass

        # Waiting for background thread to quit
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        #print(self.batch)
        with torch.cuda.stream(self.stream):
            for k in range(2):
                if isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k].to(
                        device=self.local_rank, non_blocking=True
                    )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(
            self.stream
        )  # wait tensor to put on GPU
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        # If the dataloader is to be freed, shutdown its BackgroundGenerator
        self._shutdown_background_thread()


@torch.no_grad()
def _momentum_update_key_encoder(encoder_q, encoder_k, m=0.0):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

class FreezeLayer(object):
    def __init__(self, model, freeze_layers=[], freeze_epochs=-1):

        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        self.model = model
        self.freeze_layers = freeze_layers
        self.freeze_epochs = freeze_epochs


    def before_step(self, epoch):
        # Freeze specific layers
        if epoch <= self.freeze_epochs:
            print("self.trainer.iter <= self.freeze_iters and not self.is_frozen")
            self.freeze_specific_layer()

        # Recover original layers status
        if epoch > self.freeze_epochs:
            print("self.trainer.iter > self.freeze_iters and self.is_frozen")
            self.open_all_layer()



    def freeze_specific_layer(self):
        for layer in self.freeze_layers:
            if not hasattr(self.model, layer):
                self._logger.info(f'{layer} is not an attribute of the model, will skip this layer')


        for name, module in self.model.named_children():
            if name not in self.freeze_layers:
                continue
            for param in module.parameters():
                #print(param.name)
                param.requires_grad = False

        # Change BN in freeze layers to eval mode
        for name, module in self.model.named_children():
            if name in self.freeze_layers: module.eval()

        self.is_frozen = True

    def open_all_layer(self):
        self.model.train()
        for name, module in self.model.named_children():
            for param in module.parameters():
                param.requires_grad = True

        self.is_frozen = False


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


# embedding_xyz_clone = torch.randn((14,12,1024))
# restruct_embedding_xyz = embedding_xyz_clone
# mask_index = sorted(random.sample(range(12),6))
# is_detach = True
# m = 2
# margin = 0.3
# hard_mining = True
def Triple_loss(restruct_embedding_xyz, embedding_xyz_clone, mask_index, m = 2, margin = 0.3, hard_mining = True, is_detach = True):
    #restruct_embedding_xyz, embedding_xyz_clone, mask_index
    assert len(mask_index) > 2*m+1
    _, len_time, _ = embedding_xyz_clone.shape
    set_orgin_all = set(range(len_time))
    set_mask_all = set(mask_index)
    if is_detach:
        anchor_t = embedding_xyz_clone.detach()[:, mask_index, None, :]
    else:
        anchor_t = embedding_xyz_clone[:, mask_index, None, :]
    # print(anchor_t.shape)

    positive_t = restruct_embedding_xyz[:, mask_index, None, :]
    # print(positive_t.shape)


    neg_list = [list(set_orgin_all - set(range(mask-m,mask+m+1))) for mask in mask_index]
    min_len = min([len(nl) for nl in neg_list])
    neg_list = [sorted(random.sample(nl, min_len)) for nl in neg_list]
    # print(neg_list)

    neg_rebuild_list = [list(set_mask_all - set(range(mask-m,mask+m+1))) for mask in mask_index]
    min_len = min([len(nl) for nl in neg_rebuild_list])
    neg_rebuild_list = [sorted(random.sample(nl, min_len)) for nl in neg_rebuild_list]
    # print(neg_rebuild_list)

    negitive_t = embedding_xyz_clone[:, neg_list, :]
    # print(negitive_t.shape)

    negitive_rebuild_t = restruct_embedding_xyz[:, neg_rebuild_list, :]
    # print(negitive_rebuild_t.shape)

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

    pos_neg = torch.cat([positive_t, negitive_t, negitive_rebuild_t], dim=-2)
    # print(pos_neg.shape)
    dist_mat = 1-cos(anchor_t, pos_neg)
    bs,l_a, l_pn = dist_mat.shape
    dist_mat = dist_mat.reshape(bs*l_a,l_pn)

    is_pos = torch.zeros_like(dist_mat)
    is_pos[:, :1] = 1
    is_neg = torch.zeros_like(dist_mat)
    is_neg[:, 1:] = 1

    # if norm_feat:
    #     dist_mat = cosine_dist(embedding, embedding)
    # else:
    #     dist_mat = euclidean_dist(embedding, embedding)


    # N = dist_mat.size(0)
    # is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    # is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss