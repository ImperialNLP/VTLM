import sys
sys.path.insert(0, "/home/sercan/anaconda3/envs/projtorch/lib/python3.6/site-packages/fastBPE-0.1.1-py3.6-linux-x86_64.egg")
import torch
import fastBPE as fp
from src.model.transformer import TransformerModel
from src.utils import AttrDict
from src.data.dictionary import Dictionary
import stanza
from src.model.transformer import *
from heapq import *
import cv2
import pickle
import fastBPE
import time
import numpy as np
from src.utils import to_cuda
from collections import OrderedDict
import os
import stanza

def fwd(model, x, lengths, causal,
        src_enc=None, src_len=None,
        positions=None, langs=None, image_langs=None, cache=None, image_feats=None, bboxes=None,
        result_dict=None, phrase_indices=None, labels=None, results_dict=None, mx_heap=None, activitiy_dict=None, im_name=None
        ):
    """
    Inputs:
        `x` LongTensor(slen, bs), containing word indices
        `lengths` LongTensor(bs), containing the length of each sentence
        `causal` Boolean, if True, the attention is only done over previous hidden states
        `positions` LongTensor(slen, bs), containing word positions
        `langs` LongTensor(slen, bs), containing language IDs
    """
    # check inputs
    slen, bs = x.size()

    assert lengths.size(0) == bs
    assert lengths.max().item() <= slen

    x = x.transpose(0, 1)  # batch size as dimension 0
    assert (src_enc is None) == (src_len is None)
    if src_enc is not None:
        assert model.is_decoder
        assert src_enc.size(0) == bs

    # generate masks
    mask, attn_mask = get_masks(slen, lengths, causal)

    if model.is_decoder and src_enc is not None:
        src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

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
    tensor = tensor + model.position_embeddings(positions).expand_as(tensor)
    if langs is not None and model.use_lang_emb:
        tensor = tensor + model.lang_embeddings(langs)
    tensor = model.layer_norm_emb(tensor)
    tensor = F.dropout(tensor, p=model.dropout, training=model.training)
    tensor *= mask.unsqueeze(-1).to(tensor.dtype)

    # langs = torch.cat((langs.cuda(), image_langs.cuda()))

    image_langs = image_langs.transpose(0, 1)

    img_mask = torch.ones([mask.shape[0], image_feats.shape[1]]).type_as(mask)
    img_attn_mask = torch.ones([mask.shape[0], image_feats.shape[1]]).type_as(mask)
    mask = torch.cat((mask, img_mask), dim=1)
    attn_mask = torch.cat((attn_mask, img_attn_mask), dim=1)
    # detection_classes
    image_regions = model.projector(image_feats)
    regional_encodings = model.regional_encodings(bboxes)
    image_regions += regional_encodings

    image_regions = image_regions + model.lang_embeddings(image_langs)
    tensor = torch.cat((tensor, image_regions), dim=1)


    # transformer layers
    for i in range(model.n_layers):
        total_attns = None

        # self attention
        attn, weights = model.attentions[i](tensor, attn_mask, cache=cache)
        

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
                    total_attns += box_attns[head,inds,:].mean(0).sum(0)


            total_attns /= bboxes.shape[1]
            acc = (preds[head] * labels).nonzero(as_tuple=True)[0].shape[0] / preds[head].shape[0]
            if "layer:" + str(i) + " " + "head:" + str(head) not in results_dict:
                results_dict["layer:" + str(i) + " " + "head:" + str(head)] = acc
                activitiy_dict["layer:" + str(i) + " " + "head:" + str(head)] = total_attns.item()
            else:
                results_dict["layer:" + str(i) + " " + "head:" + str(head)] += acc
                activitiy_dict["layer:" + str(i) + " " + "head:" + str(head)] += total_attns.item()
            heappush(h["layer:" + str(i) + " " + "head:" + str(head)], (acc, im_name))

        # print(preds,labels)
        tensor = tensor + attn
        tensor = model.layer_norm1[i](tensor)

        # FFN
        if ('%i_in' % i) in model.memories:
            tensor = tensor + model.memories['%i_in' % i](tensor)
        else:
            tensor = tensor + model.ffns[i](tensor)
        tensor = model.layer_norm2[i](tensor)

        # memory
        if ('%i_after' % i) in model.memories:
            tensor = tensor + model.memories['%i_after' % i](tensor)
        # TODO: add extra layer norm here?

        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

    # update cache length
    if cache is not None:
        cache['slen'] += tensor.size(1)

    # move back sequence length to dimension 0
    tensor = tensor.transpose(0, 1)
    return tensor


def find_pred_indices(word_ids, x, visited):
    pred_indices = []
    temp = []
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

if __name__ == "__main__":

    head_words = open("head_words.txt","w+")
    reloaded = torch.load("/data/shared/ConceptualCaptions/menekse_log/menekse-img-pretrain/periodic-30.pth")
    #reloaded = torch.load("/data/shared/ConceptualCaptions/ozan_xlm_en_img/6z1z5i2akx/periodic-30.pth")
    model_params = AttrDict(reloaded['params'])
    nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True)
    activity_dict = {}
    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    print(reloaded.keys())
    model = TransformerModel(model_params, dico, is_encoder=True, with_output=True,return_att_weights=True).cuda().eval()
    model.load_state_dict(reloaded['model'])
    print(model)
    word2id = dico.word2id
    id2word = dico.id2word


    model = model.eval()
    codes_path = "/data/shared/ConceptualCaptions/concap_data_bpe50k_v2/codes"
    vocab_path = "/data/shared/ConceptualCaptions/concap_data_bpe50k_v2/vocab"
    #codes_path = "/home/sercan/new_remote/XLM-master/data/concap/30k_en/codes"
    #vocab_path = "/home/sercan/new_remote/XLM-master/data/concap/30k_en/vocab"
    bpe = fastBPE.fastBPE(codes_path, vocab_path)
    feat_path = "/home/sercan/new_remote/XLM-master/data/flickr30k-features/"
    h = {}
    for i in range(model.n_layers):
        for j in range(model.n_heads):
            h["layer:" + str(i) + " " + "head:" + str(j)] = []

    max_inst = 100
    ctr = 0
    results_dict = {}
    activity_dict = {}
    total_sentences_processed = 0
    total_regions_processed = 0
    for fname in os.listdir(feat_path):
            ctr += 1

            start = time.time()
            with open(feat_path + fname, "rb") as f:
                feat = pickle.load(f)

            im_name = fname
            annots = feat["annotations"]

            for i, asent in enumerate(feat["sents"]):
                try:


                    im_name = fname
                    im_name = im_name + "|" + str(i)

                    boxes_to_take = []
                    bboxes = []
                    phrases_to_predict = []
                    sent = bpe.apply([asent["sentence"].lower()])[0]
                    # word = bpe.apply(["cowboy hate"])[0]
                    sent_ids = torch.LongTensor(
                        [model_params.eos_index] + [word2id[z] for z in sent.split()] + [model_params.eos_index]).unsqueeze(1)
                    # word_ids = torch.LongTensor([word2id[i] for i in word.split()])
                    total_box = 0
                    all_boxes = []
                    labels = OrderedDict()
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

                            phrases_to_predict.append(phrase)

                    one_hot_labels = torch.zeros(len(phrases_to_predict), len(boxes_to_take))
                    for key in labels:
                        for bx in labels[key]:
                            one_hot_labels[key][bx] = 1

                    for qt,phz in enumerate(phrases_to_predict):
                        x = nlp(phz)
                        head_words.write("phrase is: " + phz + " head word is: ")
                        for sent in x.sentences:
                            for word in sent.words:
                                if word.deprel == "root":
                                    phrases_to_predict[qt] = word.text.lower()
                                    head_words.write(word.text + "\n")

                    phrases_bped = bpe.apply(phrases_to_predict)
                    phrases_bped = [phr.split() for phr in phrases_bped]

                    phrase_ids = []
                    if boxes_to_take == []:
                        continue
                    image_feats = torch.from_numpy(feat["feats"][np.array(boxes_to_take)])

                    langs = sent_ids.new(sent_ids.shape[0], 1).fill_(model_params.lang2id["en"]).long()
                    image_langs = image_feats.new(image_feats.shape[0], 1).fill_(model_params.lang2id["img"]).long()

                    image_regions = np.expand_dims(np.array(bboxes, dtype=np.float32), 0)
                    lengths = torch.LongTensor([sent_ids.shape[0]])
                    visited = {}

                    phrase_ids = [np.array([word2id[qt] for qt in zt]) for zt in phrases_bped]
                    phrase_indices = [torch.LongTensor(np.array(find_pred_indices(phr, sent_ids, visited))) for phr in
                                      phrase_ids]

                    x, lengths, image_feats, image_langs, langs, labels = to_cuda(sent_ids, lengths, image_feats, image_langs,
                                                                                  langs, one_hot_labels)
                    out = fwd(model, x, lengths, causal=False, langs=langs, image_langs=image_langs,
                              image_feats=np.expand_dims(feat["feats"][np.array(boxes_to_take)], 0),
                              bboxes=image_regions, phrase_indices=phrase_indices
                              , labels=one_hot_labels, results_dict=results_dict, mx_heap=h, im_name=im_name,activitiy_dict=activity_dict)
                except Exception as e:
                    print(e)
                total_sentences_processed += 1
                total_regions_processed += len(phrases_to_predict)
    for i in range(model.n_layers):
        for j in range(model.n_heads):
            results_dict["layer:" + str(i) + " " + "head:" + str(j)] /= total_sentences_processed
            activity_dict["layer:" + str(i) + " " + "head:" + str(j)] /= total_sentences_processed
    res_file = open("results_multi.pkl","wb")
    average_regions = total_regions_processed/ total_sentences_processed
    random_acc = 1 / average_regions
    print(random_acc,"random acc")
    print(results_dict)
    print(activity_dict)
    pickle.dump(results_dict,res_file)
    res_file.close()
    heap_file = open("heap_multi.pkl","wb")
    pickle.dump(h,heap_file)
    heap_file.close()
    head_words.close()
