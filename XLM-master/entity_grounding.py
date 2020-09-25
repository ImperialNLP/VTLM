#!/usr/bin/env python

import os
import sys
import pickle
import string
from pathlib import Path
from heapq import heappush
from collections import OrderedDict, defaultdict

from src.utils import AttrDict, to_cuda
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel, get_masks

import numpy as np

import torch
import torch.nn.functional as F

PUNCS = tuple(list(string.punctuation))


FAST = "/home/sercan/anaconda3/envs/projtorch/lib/python3.6/site-packages/fastBPE-0.1.1-py3.6-linux-x86_64.egg"

if os.path.exists(FAST):
    # Sercan
    sys.path.insert(0, FAST)
    CODES_PATH = "/data/shared/ConceptualCaptions/concap_data_bpe50k_v2/codes"
    VOCAB_PATH = "/data/shared/ConceptualCaptions/concap_data_bpe50k_v2/vocab"
else:
    # Ozan
    CODES_PATH = "/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/codes"
    VOCAB_PATH = "/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe/vocab"

import fastBPE


def fwd(model, x, lengths, causal, params,
        positions=None, langs=None, image_langs=None, cache=None,
        image_feats=None, bboxes=None, result_dict=None, phrase_indices=None,
        labels=None, results_dict=None, mx_heap=None,
        activity_dict=None, im_name=None):

    x = x.transpose(0, 1)  # batch size as dimension 0

    # check inputs (target word indices)
    bs, slen = x.size()

    assert lengths.size(0) == bs
    assert lengths.max().item() <= slen
    assert not model.is_decoder

    # generate masks
    mask, attn_mask = get_masks(slen, lengths, causal)

    # positions
    if positions is None:
        positions = x.new(slen).long()
        positions = torch.arange(slen, out=positions).unsqueeze(0)
    else:
        positions = positions.transpose(0, 1)

    # langs
    if langs is not None:
        langs = langs.transpose(0, 1)

    # do not recompute cached elements
    if cache is not None:
        _slen = slen - cache['slen']
        x = x[:, -_slen:]
        positions = positions[:, -_slen:]
        if langs is not None:
            langs = langs[:, -_slen:]
        mask = mask[:, -_slen:]
        attn_mask = attn_mask[:, -_slen:]

    tensor = model.embeddings(x)

    if params.scale_emb:
        # scale embeddings w.r.t pos embs
        tensor.mul_(tensor.size(-1) ** 0.5)

    # Prepare linguistic tensor
    tensor = tensor + model.position_embeddings(positions).expand_as(tensor)
    if langs is not None and model.use_lang_emb:
        tensor = tensor + model.lang_embeddings(langs)
    tensor = model.layer_norm_emb(tensor)
    tensor = F.dropout(tensor, p=model.dropout, training=model.training)
    tensor *= mask.unsqueeze(-1).to(tensor.dtype)

    # Prepare image tensor
    feats = model.projector(image_feats) + model.regional_encodings(bboxes) + \
            model.lang_embeddings(image_langs.t())

    # Apply visual dropout if any
    if params.visual_dropout > 0:
        feats = F.dropout(feats, p=params.visual_dropout, training=model.training)

    # Image masks
    img_mask = torch.ones([mask.shape[0], image_feats.shape[1]]).type_as(mask)
    img_attn_mask = img_mask.clone()

    # concatenate: put image sequence first
    if params.visual_first:
        mask = torch.cat((img_mask, mask), dim=1)
        attn_mask = torch.cat((img_attn_mask, attn_mask), dim=1)
        tensor = torch.cat((feats, tensor), dim=1)
    else:
        # TODO: this may be error-prone!
        mask = torch.cat((mask, img_mask), dim=1)
        attn_mask = torch.cat((attn_mask, img_attn_mask), dim=1)
        tensor = torch.cat((tensor, feats), dim=1)

    # transformer layers
    for i in range(model.n_layers):
        total_attns = None

        # self attention
        attn, weights = model.attentions[i](tensor, attn_mask, cache=cache)

        # TODO: These should probably change with `params.visual_first` !
        box_attns = weights[..., lengths.max().item()::][0]
        preds = torch.zeros(model.n_heads, len(phrase_indices), bboxes.shape[1])

        for head in range(model.n_heads):
            for p, inds in enumerate(phrase_indices):
                inds = inds.long()
                mx = box_attns[head, inds, :].mean(0).argmax()
                preds[head, p, mx] = 1

                if total_attns is None:
                    total_attns = box_attns[head, inds, :].mean(0).sum(0)
                else:
                    total_attns += box_attns[head, inds, :].mean(0).sum(0)

            total_attns /= bboxes.shape[1]
            acc = (preds[head] * labels).nonzero(as_tuple=True)[0].shape[0] / preds[head].shape[0]

            key = f"layer:{i} head:{head}"
            results_dict[key] += acc
            activity_dict[key] += total_attns.item()
            heappush(mx_heap[key], (acc, im_name))

        tensor = tensor + attn
        tensor = model.layer_norm1[i](tensor)
        tensor = tensor + model.ffns[i](tensor)
        tensor = model.layer_norm2[i](tensor)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

    # update cache length
    if cache is not None:
        cache['slen'] += tensor.size(1)


def find_pred_indices(word_ids, x, visited):
    temp = []
    pred_indices = []
    for i, ind in enumerate(x.transpose(1, 0)[0]):
        if ind.item() != word_ids[len(temp)]:
            temp = []
            continue
        temp.append(i)
        if len(temp) == word_ids.shape[0]:
            if str(temp) not in visited:
                pred_indices = pred_indices + temp
                visited[str(temp)] = True
                break
            temp = []
    return pred_indices


def main():
    try:
        pth = sys.argv[1]
        feat_path = sys.argv[2]
    except IndexError:
        print(f'Usage: {sys.argv[0]} <.pth checkpoint file> <features path>')
        sys.exit(1)

    # fastBPE files
    bpe = fastBPE.fastBPE(CODES_PATH, VOCAB_PATH)

    # Load model
    print(f'Loading checkpoint {pth}')
    reloaded = torch.load(pth)
    params = AttrDict(reloaded['params'])

    # Inject new parameters in case the models are older than code
    if 'scale_emb' not in params:
        params['scale_emb'] = False

    dico = Dictionary(
        reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    word2id = dico.word2id

    model = TransformerModel(params, dico, is_encoder=True, with_output=True).cuda()
    model.load_state_dict(reloaded['model'])
    model = model.eval()
    torch.set_grad_enabled(False)
    print(model)

    h = {}
    feats = {}
    total_regions_processed = 0
    total_sentences_processed = 0
    results_dict = defaultdict(float)
    activity_dict = defaultdict(float)

    # Init dict
    for i in range(model.n_layers):
        for j in range(model.n_heads):
            h[f"layer:{i} head:{j}"] = []

    print('Loading features...')
    for fname in Path(feat_path).glob('*.pkl'):
        with open(fname, "rb") as f:
            feats[fname.name] = pickle.load(f)

    n_feats = len(feats)

    heads = []

    for fidx, (fname, feat) in enumerate(feats.items()):
        annots = feat["annotations"]

        for i, asent in enumerate(feat["sents"]):
            bboxes = []
            all_boxes = []
            boxes_to_take = []
            phrases_to_predict = []
            labels = OrderedDict()

            im_name = f'{fname}|{i}'

            sent_tok = asent["sentence"].lower()
            sent_bpe = bpe.apply([sent_tok])[0]
            if sent_tok != sent_bpe:
                # skip segmented captions for simplifying processing
                print(f'[{fidx}/{n_feats}]   [Skipping] {sent_bpe!r}')
                continue
            else:
                print(f'[{fidx}/{n_feats}] [Processing] {sent_tok!r}')

            idxs = [params.eos_index] + [word2id[z] for z in sent_tok.split()] + [params.eos_index]
            idxs = torch.LongTensor(idxs).unsqueeze(1)

            for xes in list(annots["boxes"].values()):
                for b in xes:
                    all_boxes.append(b)

            for z, phrase_d in enumerate(asent["phrases"]):
                phrase = phrase_d["phrase"]
                phrase_id = phrase_d["phrase_id"]
                if phrase_id in annots["boxes"]:
                    for box in annots["boxes"][phrase_id]:
                        boxes_to_take.append(all_boxes.index(box))
                        bboxes.append(box)

                        if len(phrases_to_predict) not in labels:
                            labels[len(phrases_to_predict)] = [len(boxes_to_take) - 1]
                        else:
                            labels[len(phrases_to_predict)].append(len(boxes_to_take) - 1)

                    # append head noun
                    words = phrase.lower().split()
                    # sometimes there's a punctuation at end..
                    head = words[-1] if not words[-1].endswith(PUNCS) else words[-2]
                    # check again
                    assert head[-1] not in '.,;!?'
                    # use it
                    phrases_to_predict.append(head)

            if len(boxes_to_take) == 0:
                continue

            # n_phrases X box indicators
            one_hot_labels = torch.zeros(
                len(phrases_to_predict), len(boxes_to_take))
            for key in labels:
                for bx in labels[key]:
                    one_hot_labels[key][bx] = 1



            # bbox features
            image_feats = np.expand_dims(feat["feats"][np.array(boxes_to_take)], 0)

            # bbox coordinates
            image_regions = np.expand_dims(np.array(bboxes, dtype=np.float32), 0)

            # FIXME: Correct coordinates should be passed!

            # modality embs
            langs = idxs.new(idxs.shape[0], 1).fill_(params.lang2id["en"]).long()
            image_langs = torch.empty((image_feats.shape[1], 1)).fill_(params.lang2id['img']).long()
            lengths = torch.LongTensor([idxs.shape[0]])

            x, lengths, image_langs, langs, labels = to_cuda(
                idxs, lengths, image_langs, langs, one_hot_labels)

            phrases_to_predict = [phr.split() for phr in phrases_to_predict]

            visited = {}
            phrase_ids = [np.array([word2id[qt] for qt in zt]) for zt in phrases_to_predict]
            phrase_indices = [torch.LongTensor(np.array(find_pred_indices(phr, idxs, visited))) for phr in
                              phrase_ids]

            ###################
            # model entry point
            ###################
            fwd(model, x, lengths, causal=False, params=params,
                langs=langs, image_langs=image_langs,
                image_feats=image_feats, bboxes=image_regions,
                phrase_indices=phrase_indices, labels=one_hot_labels,
                results_dict=results_dict, mx_heap=h,
                im_name=im_name, activity_dict=activity_dict)

            total_sentences_processed += 1
            total_regions_processed += len(phrases_to_predict)

    # Dump results
    results_dict = dict(results_dict)
    activity_dict = dict(activity_dict)

    for i in range(model.n_layers):
        for j in range(model.n_heads):
            key = f"layer:{i} head:{j}"
            results_dict[key] /= total_sentences_processed
            activity_dict[key] /= total_sentences_processed

    random_acc = total_sentences_processed / total_regions_processed
    print('Chance-level: ', random_acc)

    res_file = open("results_multi.pkl", "wb")
    pickle.dump(results_dict, res_file)
    res_file.close()

    heap_file = open("heap_multi.pkl", "wb")
    pickle.dump(h, heap_file)
    heap_file.close()


if __name__ == "__main__":
    main()
