# Transformer Apps

This project applies Transformer-based architectures to real-world
machine translation tasks, using the `ted_hrlr_translate/pt_to_en`
dataset (Portuguese to English).

## Dataset Setup

TFDS can no longer download `ted_hrlr_translate/pt_to_en` because the
upstream archive is offline. To work around this, the dataset is hosted
separately and loaded through a custom `load_pt2en` helper.

### Steps

1. Download and extract the dataset archive into `~/.cache/ted_hrlr/`:

```bash
   curl -L -O https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/ted_hrlr_pt_to_en.tar.gz
   mkdir -p ~/.cache/ted_hrlr
   tar -xzf ted_hrlr_pt_to_en.tar.gz -C ~/.cache/ted_hrlr/
```

2. Download `setup.py` into the root of this project folder:

```bash
   curl -L -O https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/setup.py
```

3. Use `load_pt2en(split)` (imported from `setup`) anywhere you would
   have used `tfds.load('ted_hrlr_translate/pt_to_en', split=split,
   as_supervised=True)`. The return type is identical: a
   `tf.data.Dataset` of `(pt, en)` `tf.string` pairs.

## Tasks

### 0. Dataset

`0-dataset.py` contains the `Dataset` class, which loads and preps the
`ted_hrlr_translate/pt_to_en` dataset for machine translation.

* `data_train`: the training split, loaded via `load_pt2en('train')`
* `data_valid`: the validation split, loaded via
  `load_pt2en('validation')`
* `tokenizer_pt`: a Portuguese sub-word tokenizer, fine-tuned from
  `neuralmind/bert-base-portuguese-cased` on the training set
* `tokenizer_en`: an English sub-word tokenizer, fine-tuned from
  `bert-base-uncased` on the training set

Both tokenizers are trained with a maximum vocabulary size of `2**13`.

#### Usage

```bash
./0-main.py
```

## Requirements

* Python 3.9
* TensorFlow
* `transformers` (Hugging Face)
* `setup.py` and the extracted dataset archive as described above
