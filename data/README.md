# Data preparation

## Pre-training

- [Download]() the Conceptual Captions (CC) file that includes an additional column for the automatic German translations of captions. Place the file under `./conceptual_captions/`.
- You need the raw image files from the [Conceptual Captions (CC) dataset](https://ai.google.com/research/ConceptualCaptions) so that you can extract the regional features. When we began experimenting for VTLM, 3,030,007 images of 3,318,333 total were accessible through the URLs provided. Therefore, we discarded the remaining captions from pre-training. You need to download the images manually and store them under `./conceptual_captions/images`.
