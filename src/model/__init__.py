# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel  # , TRANSFORMER_LAYER_PARAMS


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.region_pred < 1
    assert 0 <= params.sample_alpha < 1

    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    s = params.region_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.region_mask = s[0]
    params.region_keep = s[1]
    params.region_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
    Build model.
    """
    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            model.load_state_dict(reloaded)

        logger.info("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    else:
        # build
        encoder = TransformerModel(params, dico, is_encoder=True, with_output=params.encoder_output)
        decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings)
            set_pretrain_emb(decoder, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
                enc_miss, enc_unexp = encoder.load_state_dict(enc_reload, strict=False)
                for name in sorted(enc_miss):
                    logger.info(f'Encoder reloading: missing parameter {name} will be randomly initialized')
                for name in sorted(enc_unexp):
                    logger.info(f'Encoder reloading: unexpected parameter {name} ignored')

            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}

                if params.init_dec_from_enc:
                    # Init missing params from encoder params
                    for i in range(params.n_layers):
                        for name in DECODER_ONLY_PARAMS:
                            if name % i not in dec_reload:
                                enc_name = name.replace('encoder_attn', 'attentions')
                                enc_name = enc_name.replace('layer_norm15', 'layer_norm1')
                                dec_reload[name % i] = enc_reload[enc_name % i]
                                logger.info(f'Reloading param {name % i} from param {enc_name % i}')
                    if params.reset_dec_output_bias:
                        logger.info('** Resetting decoder output bias')
                        dec_reload['pred_layer.proj.bias'].fill_(0.0)
                    decoder.load_state_dict(dec_reload, strict=False)
                else:
                    # load weights with strict=False to allow for missing parameters
                    dec_miss, dec_unexp = decoder.load_state_dict(dec_reload, strict=False)
                    for name in sorted(dec_miss):
                        logger.info(f'Decoder reloading: missing parameter {name} will be randomly initialized')
                    for name in sorted(dec_unexp):
                        logger.info(f'Decoder reloading: unexpected parameter {name} ignored')

            if params.freeze_encoder:
                for param in encoder.parameters():
                    param.requires_grad = False

        if params.zero_mask_emb:
            logger.info('** Zeroing [MASK] embedding')
            with torch.no_grad():
                encoder.embeddings.weight[5].fill_(0)
                decoder.embeddings.weight[5].fill_(0)

        logger.info("Encoder: {}".format(encoder))
        logger.info("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        return encoder.cuda(), decoder.cuda()
