"""LDA topic modeling on a tiny Hugging Face dataset (AG News test split, 7600 headlines).

AG News has 4 gold labels, so topic quality is checked by purity: map each document to its
dominant topic, let each topic vote for its majority label, and count the matches.

Usage:
    uv run scripts/topics.py
    uv run scripts/topics.py --topics 8 --out topics.html
"""

import argparse
from collections import Counter

import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

DATA_URL = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--topics", type=int, default=4)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--out", default="topics.html", help="pyLDAvis HTML output")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_URL)
    docs = [
        [t for t in simple_preprocess(text) if t not in STOPWORDS]
        for text in df["text"]
    ]

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(d) for d in docs]
    lda = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=args.topics,
        passes=args.passes,
        random_state=0,
    )

    dominant = [
        max(lda.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus
    ]
    votes: dict[int, Counter[int]] = {}
    for topic, label in zip(dominant, df["label"], strict=True):
        votes.setdefault(topic, Counter())[label] += 1

    for topic in range(args.topics):
        words = ", ".join(w for w, _ in lda.show_topic(topic, topn=8))
        label = LABELS[votes[topic].most_common(1)[0][0]] if topic in votes else "-"
        print(f"topic {topic} [{label:8s}] {words}")
    purity = sum(c.most_common(1)[0][1] for c in votes.values()) / len(df)
    print(f"docs={len(df)} vocab={len(dictionary)} purity={purity:.3f}")

    # default mds="pcoa" uses np.linalg.eig, which yields complex coords on numpy 2 and breaks JSON export
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, mds="mmds")
    pyLDAvis.save_html(vis, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
