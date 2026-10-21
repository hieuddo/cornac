"""DIGER on Amazon Beauty through Cornac's normal next-item API.

Run ``python examples/diger_example.py --features beauty-st5.npz`` with an
archive containing ``features`` (finite float32 matrix) and ``ids`` (raw item IDs).
Use ``--embedding openai`` for retained OpenAI embeddings. No API key is needed
for precomputed features. With no archive, generate Sentence-T5 features from
Cornac's item texts (requires sentence-transformers). These regenerated texts
and features are not the retained study assets; see the model README.
"""

import argparse

import numpy as np

import cornac
from cornac.data import FeatureModality
from cornac.datasets import amazon_review
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import DIGER
from cornac.models.diger import BEAUTY_OPENAI_CONFIG, BEAUTY_ST5_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", help="NPZ containing features and raw ids")
    parser.add_argument("--embedding", choices=("st5", "openai"), default="st5")
    parser.add_argument("--initializer", help="Optional matched initializer bundle")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.embedding == "openai" and not args.features:
        parser.error("OpenAI mode requires precomputed --features")

    data = amazon_review.load_feedback(category="beauty")
    if args.features:
        with np.load(args.features, allow_pickle=False) as archive:
            features, item_ids = archive["features"], archive["ids"].tolist()
    else:
        from sentence_transformers import SentenceTransformer
        texts, item_ids = amazon_review.load_text(category="beauty")
        encoder = SentenceTransformer("sentence-t5-base")
        features = encoder.encode(texts, batch_size=64, show_progress_bar=True)
        del encoder

    evaluation = NextItemEvaluation.leave_last_out(
        data=data, exclude_unknowns=True, verbose=True,
        item_feature=FeatureModality(features=features, ids=item_ids),
    )
    preset = BEAUTY_ST5_CONFIG if args.embedding == "st5" else BEAUTY_OPENAI_CONFIG
    model = DIGER(**preset, initializer=args.initializer,
                  device=args.device, verbose=True)
    cornac.Experiment(eval_method=evaluation, models=[model],
                      metrics=[Recall(k=5), Recall(k=10), Recall(k=20),
                               NDCG(k=5), NDCG(k=10), NDCG(k=20)]).run()


if __name__ == "__main__":
    main()
