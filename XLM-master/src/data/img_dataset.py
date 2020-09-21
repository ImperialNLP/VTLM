from logging import getLogger
import numpy as np
import torch
import os
import pickle


from .dataset import ParallelDataset, Dataset


logger = getLogger()


def load_images(self, sentence_ids, feat_path, img_names, n_regions):
    img_data = []

    for idx in sentence_ids:
        # Everything should be loadable. If features do not exist
        # use the dummy empty_feats.pkl
        f_name = os.path.join(feat_path, img_names[idx])
        with open(f_name, "rb") as f:
            x = pickle.load(f)
            assert len(x) != 0 and len(x["detection_scores"]) == 36
            # reduce to requested # of regions
            img_data.append({k: x[k][:n_regions] for k in x})

    return img_data


class DatasetWithRegions(Dataset):

    def __init__(self, sent, pos, image_names, masked_tokens,
                 masked_object_labels, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        self.region_features_path = params.region_feats_path
        self.image_names = np.array(image_names)
        if masked_tokens is not None and masked_object_labels is not None:
            self.masked_tokens = np.array(masked_tokens)
            self.masked_object_labels = np.array(masked_object_labels)
        else:
            self.masked_tokens = None
            self.masked_object_labels = None

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # Set RNG
        self._rng = np.random.RandomState(seed=params.iter_seed)

        # sanity checks
        self.check()

    def remove_long_sentences(self, max_len):
        indices = super().remove_long_sentences(max_len)
        self.image_names = self.image_names[indices]
        self.check()

    def select_data(self, a, b):
        super().select_data(a, b)
        self.image_names = self.image_names[a:b]

    def batch_images(self, images, feat_type):
        feats = torch.from_numpy(
            np.array([img[feat_type][:self.num_of_regions] for img in images]))
        lengths = torch.full((len(images),), self.num_of_regions, dtype=torch.long)
        return feats, lengths

    def load_images(self, sentence_ids):
        return load_images(
            sentence_ids, self.region_features_path, self.image_names,
            self.num_of_regions)

    def get_batches_iterator(self, batches, return_indices):
        masked_tokens, masked_object_id = None, None
        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                self._rng.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]

            pos = self.pos[sentence_ids]
            sent = self.batch_sentences([self.sent[a:b] for a, b in pos])

            # Visual features dictionary
            img_data = self.load_images(sentence_ids)

            if self.masked_tokens is not None and self.masked_object_labels is not None:
                masked_tokens = self.masked_tokens[sentence_ids]
                masked_object_id = self.masked_object_labels[sentence_ids]

            if return_indices:
                yield (sent, img_data, masked_tokens, masked_object_id, sentence_ids)
            else:
                yield (sent, img_data, masked_tokens, masked_object_id)


class ParallelDatasetWithRegions(ParallelDataset):
    def __init__(self, sent1, pos1, sent2, pos2, image_names, masked_tokens, masked_object_labels, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.image_names = np.array(image_names)
        self.region_features_path = params.region_feats_path
        self.num_of_regions = params.num_of_regions
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        if masked_tokens is not None and masked_object_labels is not None:
            self.masked_tokens = np.array(masked_tokens)
            self.masked_object_labels = np.array(masked_object_labels)
        else:
            self.masked_tokens = None
            self.masked_object_labels = None

        # Set RNG
        self._rng = np.random.RandomState(seed=params.iter_seed)

        # sanity checks
        self.check()

    def remove_long_sentences(self, max_len):
        indices = super().remove_long_sentences(max_len)
        self.image_names = self.image_names[indices]

    def select_data(self, a, b):
        super().select_data(a, b)
        self.image_names = self.image_names[a:b]

    def batch_images(self, images, feat_type):
        feats = torch.from_numpy(
            np.array([img[feat_type][:self.num_of_regions] for img in images]))
        lengths = torch.full((len(images),), self.num_of_regions, dtype=torch.long)
        return feats, lengths

    def load_images(self, sentence_ids):
        return load_images(
            sentence_ids, self.region_features_path, self.image_names,
            self.num_of_regions)

    def get_batches_iterator(self, batches, return_indices):
        masked_tokens, masked_object_id = None, None
        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                self._rng.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]

            # Textual stream
            pos1 = self.pos1[sentence_ids]
            pos2 = self.pos2[sentence_ids]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])

            # Visual features dictionary
            img_data = self.load_images(sentence_ids)

            if self.masked_tokens is not None and self.masked_object_labels is not None:
                masked_tokens = self.masked_tokens[sentence_ids]
                masked_object_id = self.masked_object_labels[sentence_ids]

            if return_indices:
                yield (sent1, sent2, img_data, masked_tokens, masked_object_id, sentence_ids)
            else:
                yield (sent1, sent2, img_data, masked_tokens, masked_object_id)
