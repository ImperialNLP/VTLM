# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch
import os
import pickle
import bz2

logger = getLogger()


class StreamDataset(object):

    def __init__(self, sent, pos, bs, params):
        """
        Prepare batches for data iterator.
        """
        bptt = params.bptt
        self.eos = params.eos_index

        # checks
        assert len(pos) == (sent == self.eos).sum()
        assert len(pos) == (sent[pos[:, 1]] == self.eos).sum()

        n_tokens = len(sent)
        n_batches = math.ceil(n_tokens / (bs * bptt))
        t_size = n_batches * bptt * bs

        buffer = np.zeros(t_size, dtype=sent.dtype) + self.eos
        buffer[t_size - n_tokens:] = sent
        buffer = buffer.reshape((bs, n_batches * bptt)).T
        self.data = np.zeros((n_batches * bptt + 1, bs), dtype=sent.dtype) + self.eos
        self.data[1:] = buffer

        self.bptt = bptt
        self.n_tokens = n_tokens
        self.n_batches = n_batches
        self.n_sentences = len(pos)
        self.lengths = torch.LongTensor(bs).fill_(bptt)

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return self.n_sentences

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        if not (0 <= a < b <= self.n_batches):
            logger.warning("Invalid split values: %i %i - %i" % (a, b, self.n_batches))
            return
        assert 0 <= a < b <= self.n_batches
        logger.info("Selecting batches from %i to %i ..." % (a, b))

        # sub-select
        self.data = self.data[a * self.bptt:b * self.bptt]
        self.n_batches = b - a
        self.n_sentences = (self.data == self.eos).sum().item()

    def get_iterator(self, shuffle, subsample=1):
        """
        Return a sentences iterator.
        """
        indexes = (np.random.permutation if shuffle else range)(self.n_batches // subsample)
        for i in indexes:
            a = self.bptt * i
            b = self.bptt * (i + 1)
            yield torch.from_numpy(self.data[a:b].astype(np.int64)), self.lengths


class Dataset(object):

    def __init__(self, sent, pos, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # # remove empty sentences
        # self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos) == (self.sent[self.pos[:, 1]] == eos).sum()  # check sentences indices
        # assert self.lengths.min() > 0                                     # check empty sentences

    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.eos_index
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths


    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos = self.pos[a:b]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # re-index
        min_pos = self.pos.min()
        max_pos = self.pos.max()
        self.pos -= min_pos
        self.sent = self.sent[min_pos:max_pos + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos = self.pos[sentence_ids]
            sent = [self.sent[a:b] for a, b in pos]
            sent = self.batch_sentences(sent)
            yield (sent, sentence_ids) if return_indices else sent

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, seed=None, return_indices=False):
        """
        Return a sentences iterator.
        """
        assert seed is None or shuffle is True and type(seed) is int
        rng = np.random.RandomState(seed)
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True

        # sentence lengths
        lengths = self.lengths + 2

        # select sentences to iterate over
        if shuffle:
            indices = rng.permutation(len(self.pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            rng.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)


class ParallelDataset(Dataset):

    def __init__(self, sent1, pos1, sent2, pos2, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # check number of sentences
        #assert len(self.pos1) == (self.sent1 == self.eos_index).sum()
        #assert len(self.pos2) == (self.sent2 == self.eos_index).sum()

        # remove empty sentences
        self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos1) == len(self.pos2) > 0                          # check number of sentences
        assert len(self.pos1) == (self.sent1[self.pos1[:, 1]] == eos).sum()  # check sentences indices
        assert len(self.pos2) == (self.sent2[self.pos2[:, 1]] == eos).sum()  # check sentences indices
        assert eos <= self.sent1.min() < self.sent1.max()                    # check dictionary indices
        assert eos <= self.sent2.min() < self.sent2.max()                    # check dictionary indices
        assert self.lengths1.min() > 0                                       # check empty sentences
        assert self.lengths2.min() > 0                                       # check empty sentences

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos1)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos1 = self.pos1[a:b]
        self.pos2 = self.pos2[a:b]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # re-index
        min_pos1 = self.pos1.min()
        max_pos1 = self.pos1.max()
        min_pos2 = self.pos2.min()
        max_pos2 = self.pos2.max()
        self.pos1 -= min_pos1
        self.pos2 -= min_pos2
        self.sent1 = self.sent1[min_pos1:max_pos1 + 1]
        self.sent2 = self.sent2[min_pos2:max_pos2 + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos1 = self.pos1[sentence_ids]
            pos2 = self.pos2[sentence_ids]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])
            yield (sent1, sent2, sentence_ids) if return_indices else (sent1, sent2)

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, return_indices=False):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.lengths1 + self.lengths2 + 4

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)

class ParallelDatasetWithRegions(Dataset):

    def __init__(self, sent1, pos1, sent2, pos2, image_names, masked_tokens, masked_object_labels, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size
        # self.image_feats = np.array(image_feats)
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.image_names = np.array(image_names)
        self.region_features_path = params.region_feats_path
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.masked_tokens = np.array(masked_tokens)
        self.masked_object_labels = np.array(masked_object_labels)

        # # check number of sentences
        # assert len(self.pos1) == (self.sent1 == self.eos_index).sum()
        # assert len(self.pos2) == (self.sent2 == self.eos_index).sum()

        # remove empty sentences
        self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)


    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos1) == len(self.pos2) > 0                          # check number of sentences
        assert len(self.pos1) == (self.sent1[self.pos1[:, 1]] == eos).sum()  # check sentences indices
        assert len(self.pos2) == (self.sent2[self.pos2[:, 1]] == eos).sum()  # check sentences indices
        assert eos <= self.sent1.min() < self.sent1.max()                    # check dictionary indices
        assert eos <= self.sent2.min() < self.sent2.max()                    # check dictionary indices
        assert self.lengths1.min() > 0                                       # check empty sentences
        assert self.lengths2.min() > 0                                       # check empty sentences

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.image_names = self.image_names[indices]

        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.image_names = self.image_names[indices]

        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos1)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos1 = self.pos1[a:b]
        self.pos2 = self.pos2[a:b]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.image_feats = self.image_feats[a:b]

        # re-index
        min_pos1 = self.pos1.min()
        max_pos1 = self.pos1.max()
        min_pos2 = self.pos2.min()
        max_pos2 = self.pos2.max()
        self.pos1 -= min_pos1
        self.pos2 -= min_pos2
        self.sent1 = self.sent1[min_pos1:max_pos1 + 1]
        self.sent2 = self.sent2[min_pos2:max_pos2 + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool
        masked_tokens, masked_object_id = None, None
        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]

            image_name_with_indices = zip(sentence_ids, self.image_names[sentence_ids])
            image_features, good_indices = self.load_images(self.region_features_path, image_name_with_indices)
            image_scores = self.batch_images(image_features, feat_type="detection_scores")
            img_all_dict = image_features
            # img_all_dict = self.image_feats[sentence_ids]
            # print("length of image dict: ", len(img_all_dict))
            pos1 = self.pos1[good_indices]
            pos2 = self.pos2[good_indices]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])
            if self.masked_tokens is not None and self.masked_object_labels is not None:
                masked_tokens = self.masked_tokens[good_indices]
                masked_object_id = self.masked_object_labels[good_indices]
            yield (sent1, sent2, image_scores, img_all_dict, masked_tokens, masked_object_id, sentence_ids) if return_indices else (sent1, sent2,
                                                                                                     image_scores,
                                                                                                     img_all_dict,masked_tokens, masked_object_id)

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, return_indices=False):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.lengths1 + self.lengths2 + 4

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)

    def batch_images(self, images, feat_type):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s[feat_type]) for s in images])
        imgs = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        # imgs[0] = self.bor_index
        for i, img in enumerate(images):
            feat = img[feat_type]
            try:
                imgs[0:lengths[i], i].copy_(torch.from_numpy(feat.astype(np.int64)))
                # imgs[lengths[i] - 1, i] = self.eor_index
            except Exception as e:
                print(feat.shape)
                print(lengths[i], len(lengths[i]))
                print(e)
        return imgs, lengths

    def load_images(self, region_features_path, image_name_with_indices):

        img_data = []
        good_indices = []
        for ind, image_name in image_name_with_indices:
            try:
                f_name = os.path.join(region_features_path, image_name)
                with open(f_name, "rb") as f:
                    x = pickle.load(f)
                    if len(x) != 0:
                        img_data.append(x)
                        good_indices.append(ind)
                    else:
                        with open("empty_files.txt", "a") as f:
                            f.write(f_name + "\n")
                            print('File is empty')
            except:
                    pass

        return img_data, good_indices
