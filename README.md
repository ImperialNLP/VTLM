# VTLM

[(Read the paper (EACL 2021))](https://arxiv.org/pdf/2101.10044.pdf)

Pre-trained language models have been shown to substantially improve performance in many natural language tasks. Although the early focus of such models was single language pre-training, recent advances have resulted in cross-lingual and visual pre-training meth- ods. In this paper, we combine these two approaches to learn visually-grounded cross-lingual representations. Specifically, we extend the [translation language modelling](https://github.com/facebookresearch/XLM) (Lample and Conneau, 2019) with masked region classification, and perform pre-training with three-way parallel vision & language corpora. We show that when fine-tuned for multimodal machine translation, these models obtain state-of-the-art performance. We also provide qualitative insights into the usefulness of the learned grounded representations.

```
@inproceedings{caglayan-etal-2021-cross-lingual,
    title = {{Cross-lingual Visual Pre-training for Multimodal Machine Translation}},
    author = "Caglayan, Ozan and
              Kuyu, Menekse and
              Amac, Mustafa Sercan and
              Madhyastha, Pranava and
              Erdem, Erkut and
              Erdem, Aykut and
              Specia, Lucia",
    booktitle = "Proceedings of the 16th Conference of the {E}uropean Chapter of the Association for Computational Linguistics: Short Papers",
    month = apr,
    year = "2021",
    address = "online",
    publisher = "Association for Computational Linguistics",
}
```

## About the codebase
- The codebase is a revised, improved and extended version of [XLM](https://github.com/facebookresearch/XLM).
- No changes applied to multi-GPU code, which did not work well for us. All models were trained on a single GPU.

## Datasets

### Pre-training

You need the raw image files from the [Conceptual Captions (CC) dataset](https://ai.google.com/research/ConceptualCaptions) so that you can extract the regional features. When we began experimenting for VTLM, 3,030,007 images of 3,318,333 total were accessible through the URLs provided. Therefore, we discarded the remaining captions from pre-training. You need to download the images manually and store them under `data/conceptual_captions/images`.
