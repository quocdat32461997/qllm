qllm - Quantization with LLM

## Amazon 2014 data pipelines

Two pipelines share one item universe so Semantic-IDs line up with the items that
appear in user sessions.

1. **Download** per-category metadata + 5-core reviews from Stanford SNAP:

   ```bash
   python download_amazon_2014.py --out ./amazon2014_meta \
       Beauty Sports_and_Outdoors Toys_and_Games
   ```

   Downloads both `meta_<Category>.json.gz` and `reviews_<Category>_5.json.gz` by
   default (`--meta` / `--reviews` to narrow).

2. **Preprocess (P5-style sequential data)** — reproduces the P5 notebook's
   processing (5-core filtering, time-ordered user→item sequences, 1-indexed id maps,
   item attributes, 99 negatives/user):

   ```bash
   python preprocess_amazon_2014.py Beauty Sports_and_Outdoors Toys_and_Games \
       --data-dir ./amazon2014_meta --out ./amazon2014_seq
   ```

   Writes, per category, `sequential_data.txt`, `datamaps.json`,
   `item2attributes.json`, `negative_samples.txt` into `amazon2014_seq/<Category>/`.

   On RunPod (or any fresh box), `bash preprocess.sh` runs the download + preprocess
   steps together; see the script header for `CATEGORIES` / `DATA_DIR` / `OUT_DIR` /
   `SEED` / `SAMPLE_TYPE` / `SKIP_DOWNLOAD` overrides.

   ### Preprocessed dataset statistics

   Output of the preprocessing (5-core, `seed=2022`) for the three configured
   categories. Counts match the P5 paper's published 5-core numbers, and every 5-core
   item is matched 1:1 in the metadata:

   | Category | Users | Items | Interactions | Avg len | Attributes | Meta matched |
   | --- | ---: | ---: | ---: | ---: | ---: | :---: |
   | Beauty | 22,363 | 12,101 | 198,502 | 8.88 | 2,319 | 12,101/12,101 |
   | Sports_and_Outdoors | 35,598 | 18,357 | 296,337 | 8.32 | 3,821 | 18,357/18,357 |
   | Toys_and_Games | 19,412 | 11,924 | 167,597 | 8.63 | 1,758 | 11,924/11,924 |

   - **Users / Items / Interactions** — after iterative 5-core filtering on the review
     interactions.
   - **Avg len** — mean number of items per user sequence.
   - **Attributes** — distinct category/brand attributes kept (`attribute_core=0`).
   - **Meta matched** — 5-core items found in `meta_<Category>.json.gz` (all of them).

3. **Load** via `data_module.build_amazon_datasets(config)`:
   - **Semantic-ID (products + features)** — `dataset_version: 2014`. Set
     `amazon_2014_sequential_dir` to restrict products to the P5-filtered item set
     (aligned); omit it for the full metadata dump.
   - **Sequential (per-user sessions)** — `dataset_mode: sequential` with
     `amazon_2014_sequential_dir` set.

## Citing
@proceeding{sc2026,
  author = {SC, DN},
  title = {Can LLMs remember and anlyze attributes?},
  year = {2023}
}
