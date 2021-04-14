# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import subprocess
import pickle as pkl
from collections import OrderedDict

import numpy as np
import torch
import sacrebleu

from ..utils import to_cuda, restore_segmentation, concat_batches


logger = getLogger()


def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    sentences = []
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')

    hyps, refs = [], []

    with open(hyp) as fh, open(ref) as rh:
        for line in fh:
            hyps.append(line.strip())

        for line in rh:
            refs.append(line.strip())

        score = sacrebleu.corpus_bleu(hyps, [refs], tokenize='none').score

    return score


def kl_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    _x = x.copy()
    _x[x == 0] = 1
    return np.log(len(x)) + (x * np.log(_x)).sum()


def gini_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    B = np.cumsum(np.sort(x)).mean()
    return 1 - 2 * B


def tops(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    y = np.cumsum(np.sort(x))
    top50, top90, top99 = y.shape[0] - np.searchsorted(y, [0.5, 0.1, 0.01])
    return top50, top90, top99


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params

        # create directory to store hypotheses, and reference files for BLEU evaluation
        if self.params.is_master and any(self.params.mt_steps + self.params.mmt_steps):
            params.hyp_path = os.path.join(params.dump_path, 'hypotheses')
            subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
            self.create_reference_files()

    def print_batch(self, x):
        slen, bs = x.size()
        for i in range(bs):
            idxs = x[:, i].tolist()
            s = ' '.join([self.dico.id2word[k] for k in idxs])
            print(s)

    def get_iterator(self, data_set, lang1, lang2=None, stream=False):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        n_sentences = -1
        subsample = 1

        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1][data_set].get_iterator(
                    shuffle=False, subsample=subsample)
            else:
                if 'vmono' in self.data.keys():
                    data = self.data['vmono']
                elif 'mono' in self.data.keys():
                    data = self.data['mono']
                else:
                    raise RuntimeError('No `mono` or `vmono` in self.data')

                iterator = data[lang1][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=False,
                    n_sentences=n_sentences,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            if 'vpara' in self.data.keys():
                data = self.data['vpara']
            elif 'para' in self.data.keys():
                data = self.data['para']
            else:
                raise RuntimeError('No `para` or `vpara` in self.data')

            iterator = data[(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences
            )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def get_iterator_vlm(self, data_set, lang1, lang2=None, stream=False):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        n_sentences = -1

        if lang2 is None:
            iterator = self.data['vmono'][lang1][data_set].get_iterator(
                shuffle=False,
                group_by_size=False,
                n_sentences=n_sentences,
            )
        else:
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)

            iterator = self.data['vpara'][(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences
            )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}
        data = self.data.get('para', self.data.get('vpara', None))
        assert data is not None, "Either `para` or `vpara` should be set."

        for (lang1, lang2), v in data.items():

            assert lang1 < lang2

            for data_set in ['valid', 'test']:
                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                if params.mmt_steps:
                    for (sent1, len1), (sent2, len2), _, _ in self.get_iterator(data_set, lang1, lang2):
                        lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                        lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))
                else:
                    for (sent1, len1), (sent2, len2), _ in self.get_iterator(data_set, lang1, lang2):
                        lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                        lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

    def mask_out(self, x, lengths, rng):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.bool))

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():
            for data_set in ['valid', 'test']:
                # causal prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.clm_steps:
                    self.evaluate_clm(scores, data_set, lang1, lang2)

                # prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.mlm_steps:
                    if params.eval_vlm:
                        self.evaluate_vlm(scores, data_set, lang1, lang2)
                    else:
                        self.evaluate_mlm(scores, data_set, lang1, lang2)

                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu)

                # multimodal machine translation task
                for lang1, lang2 in params.mmt_steps:
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_mmt(scores, data_set, lang1, lang2, eval_bleu)

                # report average metrics per language
                _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                if len(_clm_mono) > 0:
                    scores['%s_clm_ppl' % data_set] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                    scores['%s_clm_acc' % data_set] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])

                _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                if len(_mlm_mono) > 0:
                    scores['%s_mlm_ppl' % data_set] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                    scores['%s_mlm_acc' % data_set] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

        return scores

    def evaluate_clm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else f"{lang1}-{lang2}"

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):
            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1
            y = x[1:].masked_select(pred_mask[:-1])
            assert pred_mask.sum().item() == y.size(0)

            # cuda
            x, lengths, positions, langs, pred_mask, y = to_cuda(x, lengths, positions, langs, pred_mask, y)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=True)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # log
        logger.info("Found %i words in %s. %i were predicted correctly." % (n_words, data_set, n_valid))

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_clm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_clm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words)
        scores[acc_name] = 100. * n_valid / n_words

    def evaluate_mlm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.encoder
        model.eval()
        model = model.module if params.multi_gpu else model

        # Deterministic masking
        rng = np.random.RandomState(0)

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else f"{lang1}_{lang2}"

        n_words = 0
        xe_loss = 0
        n_valid = 0
        y, pred_mask = None, None

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None and not self.params.only_vlm)):
            if lang2 is None:
                positions = None
                sent1, len1 = batch
                langs = sent1.clone().fill_(lang1_id) if params.n_langs > 1 else None
                x, lengths = sent1, len1
            else:
                _, (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(
                    sent1, len1, lang1_id, sent2, len2, lang2_id,
                    params.pad_index, params.eos_index, reset_positions=True)

            if params.word_pred > 0:
                # words to predict
                x, y, pred_mask = self.mask_out(x, lengths, rng)

            # NOTE: Check for visual_first
            elif params.eval_probes.startswith('drop_last:'):
                # drop last words
                option = params.eval_probes.replace('drop_last:', '')
                # can be lang1, lang2 or lang1-lang2
                drop_langs = option.split('-')

                pred_mask = torch.zeros_like(x).bool()
                bs = x.size(1)

                if lang1 in drop_langs:
                    pred_mask[len1 - 2, range(bs)] = True
                if lang2 is not None and lang2 in drop_langs:
                    pred_mask[len1 + len2 - 2, range(bs)] = True

                y = x[pred_mask]
                x.masked_scatter_(pred_mask, y.clone().fill_(params.mask_index))

            # cuda
            x, y, pred_mask, lengths, positions, langs = to_cuda(
                x, y, pred_mask, lengths, positions, langs)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_mlm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_mlm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100. * n_valid / n_words if n_words > 0 else 0.

    def evaluate_vlm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.encoder
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        img_id = params.lang2id['img']
        l1l2 = lang1 if lang2 is None else f"{lang1}_{lang2}"

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_iterator_vlm(data_set, lang1, lang2, stream=(lang2 is None)):
            if lang2 is not None:
                # vTLM
                _, (img_boxes, img_feats, img_labels), (x1, len1), (x2, len2) = batch
                image_langs = torch.empty((params.num_of_regions, len1.size(0))).long().fill_(img_id)
                x, lengths, positions, langs = concat_batches(
                    x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                    params.eos_index, reset_positions=True)
            else:
                # vMLM
                (x, len1), (img_boxes, img_feats, img_labels), _ = batch
                langs = x.clone().fill_(lang1_id)
                image_langs = torch.empty((params.num_of_regions, len1.size(0))).long().fill_(img_id)
                lengths = len1
                positions = None

            if params.word_pred > 0:
                # words to predict
                x, y, pred_mask = self.mask_out(x, lengths, rng)

            # NOTE: Check for visual_first
            elif params.eval_probes.startswith('drop_last:'):
                # drop last words
                option = params.eval_probes.replace('drop_last:', '')
                # can be lang1, lang2 or lang1-lang2
                drop_langs = option.split('-')

                pred_mask = torch.zeros_like(x).bool()
                bs = x.size(1)

                if lang1 in drop_langs:
                    pred_mask[len1 - 2, range(bs)] = True
                if lang2 is not None and lang2 in drop_langs:
                    pred_mask[len1 + len2 - 2, range(bs)] = True

                y = x[pred_mask]
                x.masked_scatter_(pred_mask, y.clone().fill_(params.mask_index))

            # cuda
            x, y, pred_mask, lengths, positions, langs, image_langs = to_cuda(
                x, y, pred_mask, lengths, positions, langs, image_langs)

            img_boxes, img_feats = to_cuda(img_boxes, img_feats)

            # forward / loss
            tensor = model(
                'fwd', x=x, lengths=lengths, positions=positions, langs=langs,
                image_langs=image_langs, causal=False,
                img_boxes=img_boxes, img_feats=img_feats)

            # Fetch linguistic part of the hidden states for accuracy computation
            if params.visual_first:
                sent_tensor = tensor[params.num_of_regions:]
            else:
                sent_tensor = tensor[:-params.num_of_regions]

            word_scores, loss = model(
                    'predict', tensor=sent_tensor,
                    pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_mlm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_mlm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100. * n_valid / n_words if n_words > 0 else 0.


class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model


class EncDecEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder

    def evaluate_mmt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        img_id = params.lang2id["img"]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # batches of info dumped from cross-attention layers
        full_cross_att = []

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator(data_set, lang1, lang2):
            # generate batch
            _, (img_boxes, img_feats, img_labels), (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)
            img_langs = torch.empty((params.num_of_regions, langs1.size(1))).long().fill_(img_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y, img_langs, img_boxes, img_feats = to_cuda(
                x1, len1, langs1, x2, len2, langs2, y, img_langs, img_boxes, img_feats)
            # encode source sentence
            enc1 = encoder(
                'fwd', x=x1, lengths=len1, langs=langs1,
                image_langs=img_langs, causal=False,
                img_boxes=img_boxes, img_feats=img_feats)

            # encode source sentence
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1

            # decode target sentence
            dec2 = decoder(
                'fwd', x=x2, lengths=len2, langs=langs2, causal=True,
                src_enc=enc1, src_len=len1 + params.num_of_regions)

            # loss
            word_scores, loss = decoder(
                'predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

            # generate translation - translate / convert to text
            if eval_bleu:
                if params.eval_max_len == -1:
                    max_len = int(1.5 * len1.max().item() + 10)
                else:
                    max_len = params.eval_max_len
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(
                        enc1, len1 + params.num_of_regions, lang2_id,
                        max_len=max_len)
                else:
                    generated, lengths = decoder.generate_beam(
                        enc1, len1 + params.num_of_regions, lang2_id,
                        max_len=max_len, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping)

                hyp = convert_to_text(generated, lengths, self.dico, params)
                hypothesis.extend(hyp)
                if len(decoder.cross_att) > 0:
                    src_sent = convert_to_text(x1, len1, self.dico, params)
                    decoder.cross_att['srcs'] = src_sent
                    decoder.cross_att['hyps'] = hyp
                    full_cross_att.append(decoder.cross_att)

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        if params.dump_att_dict:
            with open(f'{params.hyp_path}/{data_set}.pkl', 'wb') as f:
                pkl.dump(full_cross_att, f)

        # compute BLEU
        if eval_bleu:
            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mmt_bleu' % (data_set, lang1, lang2)] = bleu

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator(data_set, lang1, lang2):

            # generate batch
            _, (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1

            # decode target sentence
            dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            # loss
            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

            # generate translation - translate / convert to text
            if eval_bleu:
                if params.eval_max_len == -1:
                    max_len = int(1.5 * len1.max().item() + 10)
                else:
                    max_len = params.eval_max_len
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=max_len)
                else:
                    generated, lengths = decoder.generate_beam(
                        enc1, len1, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute BLEU
        if eval_bleu:
            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu
