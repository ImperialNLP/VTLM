#!/usr/bin/env python

import os
import sys
import pickle
import string
from pathlib import Path
from heapq import heappush
from collections import defaultdict

from src.utils import AttrDict, to_cuda
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel, get_masks

import numpy as np
import tqdm

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


def reorder_and_normalize_boxes(box_list, width, height, normcoords):
    # original order: xmin, ymin, xmax, ymax
    # required order: ymin, xmin, ymax, xmax (1, 0, 3, 2)
    new_box_list = []

    # for checking validity
    normc = ['|'.join(map(lambda x: f'{x:.5f}', xx)) for xx in normcoords]
    nc = np.array(normcoords)

    for box in box_list:
        xmin, ymin, xmax, ymax = box
        new_box_list.append(
            [ymin / height, xmin / width, ymax / height, xmax / width])
        _str = '|'.join(map(lambda x: f'{x:.5f}', new_box_list[-1]))
        assert _str in normc or \
            np.abs(nc - np.array(new_box_list[-1])).sum(1).min() < 1e-5

    return new_box_list


def fwd(model, x, lengths, causal, params,
        positions=None, langs=None, image_langs=None,
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

    tensor = model.embeddings(x)

    if model.scale_emb:
        # scale embeddings w.r.t pos embs
        tensor.mul_(tensor.size(-1) ** 0.5)

    # Prepare linguistic tensor
    txt_pos_emb = model.position_embeddings(positions).expand_as(tensor)
    lang_emb = None
    if langs is not None and model.use_lang_emb:
        lang_emb = model.lang_embeddings(langs)

    _tensor = tensor + txt_pos_emb
    if lang_emb is not None:
        _tensor += lang_emb

    tensor = model.layer_norm_emb(_tensor)
    tensor = F.dropout(tensor, p=model.dropout, training=model.training)
    tensor *= mask.unsqueeze(-1).to(tensor.dtype)

    # Prepare image tensor
    feats = model.projector(image_feats)
    img_pos_emb = model.regional_encodings(bboxes)
    img_lang_emb = model.lang_embeddings(image_langs.t())
    _feats = feats + img_pos_emb + img_lang_emb

    # is a no-op, if it was not enabled during training
    feats = model.layer_norm_vis(_feats)

    # Apply visual dropout if any
    if model.v_dropout > 0:
        feats = F.dropout(feats, p=model.v_dropout, training=model.training)

    # Image masks
    img_mask = torch.ones([mask.shape[0], image_feats.shape[1]]).type_as(mask)
    img_attn_mask = img_mask.clone()

    # concatenate: put image sequence first
    if model.visual_first:
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
        total_attns = 0.0
        preds = tensor.new_zeros((model.n_heads, len(phrase_indices), bboxes.shape[1]))

        # self attention
        attn, weights = model.attentions[i](tensor, attn_mask)
        attn = F.dropout(attn, p=model.dropout, training=model.training)
        tensor = tensor + attn
        tensor = model.layer_norm1[i](tensor)
        tensor = tensor + model.ffns[i](tensor)
        tensor = model.layer_norm2[i](tensor)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        weights.squeeze_(0)

        # slice visual part
        if model.visual_first:
            # n_heads, n_total_embs, n_visual_embs
            box_attns = weights[..., :bboxes.shape[1]]
            pos_offset = bboxes.shape[1]
        else:
            # n_heads, n_total_embs, n_visual_embs
            box_attns = weights[..., lengths[0]:]
            pos_offset = 0

        for head in range(model.n_heads):
            for p, pos in enumerate(phrase_indices):
                img_att = box_attns[head, pos + pos_offset, :]
                preds[head, p, img_att.argmax()] = 1
                total_attns += img_att.sum()

            total_attns /= bboxes.shape[1]
            acc = (preds[head] * labels).nonzero(as_tuple=True)[0].shape[0] / preds[head].shape[0]

            key = f"layer:{i} head:{head}"
            results_dict[key] += acc
            activity_dict[key] += total_attns.item()



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
    model.load_state_dict(reloaded['model'], strict=False)
    model = model.eval()
    torch.set_grad_enabled(False)

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

    for fidx, (fname, feat) in enumerate(tqdm.tqdm(feats.items())):
        annots = feat["annotations"]
        boxes = annots["boxes"]
        width, height = annots["width"], annots["height"]
        sents = feat["sents"]
        feats = feat["feats"]
        cords = feat["normalized_coords"]

        all_boxes = []
        # gather all boxes
        for xes in boxes.values():
            for b in xes:
                all_boxes.append(b)

        for i, asent in enumerate(sents):
            sent_tok = asent["sentence"].lower()
            sent_bpe = bpe.apply([sent_tok])[0]
            if sent_tok != sent_bpe:
                # skip segmented captions for simplifying processing
                continue


            bboxes = []
            boxes_to_take = []
            labels = defaultdict(list)
            phrases_to_predict = []
            im_name = f'{fname}|{i}'
            sent_words = sent_tok.split()

            for phrase_d in asent["phrases"]:
                phrase = phrase_d["phrase"]
                phrase_id = phrase_d["phrase_id"]
                # starting pos in sent_tok.split()
                first_idx = phrase_d["first_word_index"]

                if phrase_id not in boxes:
                    # no associated bbox with this phrase
                    continue

                # split phrase words
                head = phrase.lower().split()[-1]

                # sometimes there's a punctuation at end.
                if head.endswith(PUNCS):
                    continue

                for box in boxes[phrase_id]:
                    boxes_to_take.append(all_boxes.index(box))
                    bboxes.append(box)
                    labels[len(phrases_to_predict)].append(len(boxes_to_take) - 1)

                # add position offset'ed by <EOS>
                pos = sent_words[first_idx:].index(head) + first_idx + 1
                assert sent_words[pos - 1] == head
                phrases_to_predict.append((pos, head))

            if len(boxes_to_take) == 0:
                # no box found for this caption
                continue

            # bbox features
            image_feats = feats[np.array(boxes_to_take)][None]

            # bbox coordinates to pass to the model (should be normalized!)
            bboxes = reorder_and_normalize_boxes(
                bboxes, width, height, cords.tolist())
            image_regions = np.array(bboxes, dtype=np.float32)[None]

            # modality embs
            idxs = [params.eos_index] + [word2id[z] for z in sent_words] + [params.eos_index]
            idxs = torch.LongTensor(idxs).unsqueeze(1)
            langs = idxs.clone().fill_(params.lang2id["en"]).long()
            image_langs = torch.empty((image_feats.shape[1], 1)).fill_(params.lang2id['img']).long()
            lengths = torch.LongTensor([idxs.shape[0]])
            phrase_indices = [w[0] for w in phrases_to_predict]

            # n_phrases X box indicators
            one_hot_labels = torch.zeros(len(phrases_to_predict), len(boxes_to_take))
            for key in labels:
                for bx in labels[key]:
                    one_hot_labels[key][bx] = 1

            x, lengths, image_langs, langs, one_hot_labels = to_cuda(
                idxs, lengths, image_langs, langs, one_hot_labels)

            ###################
            # model entry point
            ###################
            fwd(model, x, lengths, causal=False, params=params,
                langs=langs, image_langs=image_langs,
                image_feats=image_feats, bboxes=image_regions,
                phrase_indices=phrase_indices, labels=one_hot_labels,
                results_dict=results_dict,
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
    print('Avg acc: ', np.array(list(results_dict.values())).mean())

    res_file = open("results_multi.pkl", "wb")
    pickle.dump(results_dict, res_file)
    res_file.close()

    for i in range(model.n_layers):
        accs = np.array([v for k, v in results_dict.items() if k.startswith(f'layer:{i} ')])
        acts = np.array([v for k, v in activity_dict.items() if k.startswith(f'layer:{i} ')])
        print(f'Layer {i}: avg. acc: {accs.mean():.3f} avg. act: {acts.mean():.3f}')

if __name__ == "__main__":
    main()
