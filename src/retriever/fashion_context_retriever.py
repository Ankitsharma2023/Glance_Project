from pathlib import Path
import re
import sys

import faiss
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.indexer.encoders import (
    CLIPEncoder,
    FashionCLIPEncoder,
)

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

CANDIDATE_K = 100
COVERAGE_THRESHOLD = 0.25
CONTEXT_NEGATIVE_WEIGHT = 0.50


COLORS = {
    "red",
    "blue",
    "yellow",
    "white",
    "black",
    "green",
    "grey",
    "gray",
    "brown",
    "pink",
    "orange",
    "purple",
    "beige",
    "navy",
}


GARMENT_ALIASES = {
    "shirt": {0, 1},
    "blouse": {0},
    "top": {0, 1},
    "t-shirt": {1},
    "tshirt": {1},
    "sweatshirt": {1},
    "sweater": {2},
    "cardigan": {3},
    "jacket": {4},
    "blazer": {4},
    "vest": {5},
    "pants": {6},
    "trousers": {6},
    "shorts": {7},
    "skirt": {8},
    "coat": {9},
    "raincoat": {9, 4},
    "dress": {10},
    "jumpsuit": {11},
    "cape": {12},
    "glasses": {13},
    "hat": {14},
    "tie": {16},
    "glove": {17},
    "watch": {18},
    "belt": {19},
    "stockings": {21},
    "sock": {22},
    "shoe": {23},
    "shoes": {23},
    "bag": {24},
    "wallet": {24},
    "scarf": {25},
    "umbrella": {26},
}


STYLE_TERMS = {
    "professional",
    "business",
    "formal",
    "casual",
    "weekend",
    "smart",
    "relaxed",
    "elegant",
    "sporty",
    "streetwear",
}


CONTEXT_TERMS = {
    "office",
    "park",
    "bench",
    "street",
    "city",
    "home",
    "indoors",
    "inside",
    "outdoors",
    "urban",
    "walk",
    "walking",
    "sitting",
}


CONTEXT_PROTOTYPES = {
    "office": {
        "positive": [
            "inside a modern office",
            "professional office interior",
            "corporate workplace",
            "business office environment",
            "indoor office workspace",
        ],
        "negative": [
            "outdoor street",
            "urban sidewalk",
            "fashion runway",
            "park",
            "home interior",
            "closet or dressing room",
        ],
    },
    "park": {
        "positive": [
            "in a green park",
            "outdoor public park",
            "park with trees and grass",
            "sitting in a park",
            "park bench outdoors",
        ],
        "negative": [
            "office interior",
            "home interior",
            "fashion runway",
            "urban street",
            "closet or dressing room",
        ],
    },
    "city": {
        "positive": [
            "walking in a city",
            "urban street scene",
            "city sidewalk",
            "outdoor city walk",
            "walking through an urban area",
        ],
        "negative": [
            "office interior",
            "home interior",
            "green park",
            "fashion runway",
            "closet or dressing room",
        ],
    },
    "street": {
        "positive": [
            "urban street scene",
            "walking on a street",
            "city sidewalk outdoors",
            "street fashion outdoors",
        ],
        "negative": [
            "office interior",
            "home interior",
            "green park",
            "closet or dressing room",
        ],
    },
    "home": {
        "positive": [
            "inside a home",
            "home interior",
            "domestic living space",
            "casual indoor home setting",
        ],
        "negative": [
            "office interior",
            "urban street",
            "green park",
            "fashion runway",
        ],
    },
}


def parse_query(query):
    text = query.lower()

    normalized = re.sub(
        r"[^a-z0-9\- ]+",
        " ",
        text,
    )

    tokens = normalized.split()

    fashion_clauses = []

    for index in range(len(tokens) - 1):
        color = tokens[index]
        garment = tokens[index + 1]

        if color in COLORS and garment in GARMENT_ALIASES:
            fashion_clauses.append(
                {
                    "text": f"{color} {garment}",
                    "garment": garment,
                    "category_ids": GARMENT_ALIASES[garment],
                }
            )

    captured_garments = {clause["garment"] for clause in fashion_clauses}

    for token in tokens:
        if token not in GARMENT_ALIASES:
            continue

        if token in captured_garments:
            continue

        fashion_clauses.append(
            {
                "text": token,
                "garment": token,
                "category_ids": GARMENT_ALIASES[token],
            }
        )

    style_tokens = [token for token in tokens if token in STYLE_TERMS]

    context_tokens = [token for token in tokens if token in CONTEXT_TERMS]

    return {
        "fashion_clauses": fashion_clauses,
        "style": " ".join(style_tokens),
        "context": " ".join(context_tokens),
    }


def get_context_prototypes(context_text):
    tokens = set(context_text.lower().split())

    for context_name in (
        "office",
        "park",
        "city",
        "street",
        "home",
    ):
        if context_name in tokens:
            return CONTEXT_PROTOTYPES[context_name]

    return None


def minmax(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


class FashionContextRetriever:

    def __init__(self):
        print("Loading Fashion Context Retriever...")

        self.image_mapping = pd.read_csv(FEATURE_DIR / "image_mapping.csv")

        self.fashion_index = faiss.read_index(
            str(FEATURE_DIR / "global_fashionclip.faiss")
        )

        self.clip_index = faiss.read_index(str(FEATURE_DIR / "global_clip.faiss"))

        self.clip_embeddings = np.load(FEATURE_DIR / "global_clip.npy").astype(
            np.float32
        )

        self.fashion_embeddings = np.load(
            FEATURE_DIR / "global_fashionclip.npy"
        ).astype(np.float32)

        self.region_embeddings = np.load(FEATURE_DIR / "region_fashionclip.npy").astype(
            np.float32
        )

        self.region_mapping = pd.read_csv(FEATURE_DIR / "region_mapping.csv")

        if len(self.region_embeddings) != len(self.region_mapping):
            raise RuntimeError("Region embedding/mapping mismatch.")

        self.fashion_encoder = FashionCLIPEncoder()

        self.context_encoder = CLIPEncoder()

        self.region_groups = {
            image_id: group
            for image_id, group in self.region_mapping.groupby("image_id")
        }

        print("Retriever ready.")

    def search(self, query, top_k=5):

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        parsed = parse_query(query)

        fashion_clauses = parsed["fashion_clauses"]

        fashion_query_embedding = self.fashion_encoder.encode_texts([query])

        clip_query_embedding = self.context_encoder.encode_texts([query])

        _, fashion_indices = self.fashion_index.search(
            fashion_query_embedding,
            CANDIDATE_K,
        )

        _, clip_indices = self.clip_index.search(
            clip_query_embedding,
            CANDIDATE_K,
        )

        candidate_indices = np.asarray(
            list(dict.fromkeys(fashion_indices[0].tolist() + clip_indices[0].tolist())),
            dtype=np.int64,
        )

        global_scores = (
            self.fashion_embeddings[candidate_indices] @ fashion_query_embedding[0]
        ).astype(np.float32)

        if fashion_clauses:
            clause_texts = [clause["text"] for clause in fashion_clauses]

            clause_embeddings = self.fashion_encoder.encode_texts(clause_texts)
        else:
            clause_texts = []
            clause_embeddings = None

        context_text = parsed["context"]

        if context_text:
            context_embedding = (self.context_encoder.encode_texts([context_text]))[0]
        else:
            context_embedding = None

        context_prototypes = get_context_prototypes(context_text)

        if context_prototypes is not None:
            positive_embeddings = self.context_encoder.encode_texts(
                context_prototypes["positive"]
            )

            negative_embeddings = self.context_encoder.encode_texts(
                context_prototypes["negative"]
            )
        else:
            positive_embeddings = None
            negative_embeddings = None

        style_text = parsed["style"]

        if style_text:
            style_embedding = (self.fashion_encoder.encode_texts([style_text]))[0]
        else:
            style_embedding = None

        result_rows = []

        for candidate_position, global_score in zip(
            candidate_indices,
            global_scores,
        ):

            image_row = self.image_mapping.iloc[candidate_position]

            image_id = image_row["image_id"]

            clause_scores = []

            if fashion_clauses and image_id in self.region_groups:

                image_regions = self.region_groups[image_id]

                for clause_index, clause in enumerate(fashion_clauses):

                    compatible_regions = image_regions[
                        image_regions["category_id"].isin(clause["category_ids"])
                    ]

                    if compatible_regions.empty:
                        clause_scores.append(0.0)
                        continue

                    region_indices = (
                        compatible_regions["region_index"].astype(int).to_numpy()
                    )

                    embeddings = self.region_embeddings[region_indices]

                    similarities = embeddings @ clause_embeddings[clause_index]

                    clause_scores.append(float(similarities.max()))

            elif fashion_clauses:
                clause_scores = [0.0 for _ in fashion_clauses]

            if clause_scores:
                clause_array = np.asarray(
                    clause_scores,
                    dtype=np.float32,
                )

                mean_score = float(np.mean(clause_array))

                min_score = float(np.min(clause_array))

                if len(clause_array) > 1:
                    region_score = 0.60 * mean_score + 0.40 * min_score
                else:
                    region_score = mean_score

                matched = clause_array >= COVERAGE_THRESHOLD

                coverage_score = float(np.mean(matched))

                all_matched = float(np.all(matched))

            else:
                region_score = 0.0
                coverage_score = 0.0
                all_matched = 0.0

            if context_embedding is not None:

                image_context = self.clip_embeddings[candidate_position]

                if positive_embeddings is not None:

                    positive_score = float(np.mean(positive_embeddings @ image_context))

                    negative_score = float(np.max(negative_embeddings @ image_context))

                    context_score = (
                        positive_score - CONTEXT_NEGATIVE_WEIGHT * negative_score
                    )

                else:
                    context_score = float(image_context @ context_embedding)

            else:
                context_score = 0.0

            if style_embedding is not None:
                style_score = float(
                    self.fashion_embeddings[candidate_position] @ style_embedding
                )
            else:
                style_score = 0.0

            result_rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_row["image_path"],
                    "global_score": float(global_score),
                    "region_score": region_score,
                    "coverage_score": coverage_score,
                    "all_clauses_matched": all_matched,
                    "style_score": style_score,
                    "context_score": context_score,
                    "clause_scores": clause_scores,
                }
            )

        results = pd.DataFrame(result_rows)

        results["global_norm"] = minmax(results["global_score"])

        results["region_norm"] = minmax(results["region_score"])

        if style_embedding is not None:
            results["style_norm"] = minmax(results["style_score"])
        else:
            results["style_norm"] = 0.0

        if context_embedding is not None:
            results["context_norm"] = minmax(results["context_score"])
        else:
            results["context_norm"] = 0.0

        component_scores = {"global": results["global_norm"]}

        component_weights = {"global": 0.25}

        if fashion_clauses:
            component_scores["region"] = results["region_norm"]

            component_weights["region"] = 0.30

            component_scores["coverage"] = results["coverage_score"]

            component_weights["coverage"] = 0.15

            if len(fashion_clauses) > 1:
                component_scores["conjunction"] = results["all_clauses_matched"]

                component_weights["conjunction"] = 0.10

        if style_embedding is not None:
            component_scores["style"] = results["style_norm"]

            component_weights["style"] = 0.15

        if context_embedding is not None:
            component_scores["context"] = results["context_norm"]

            component_weights["context"] = 0.15

        total_weight = sum(component_weights.values())

        results["final_score"] = 0.0

        for name, score in component_scores.items():

            weight = component_weights[name] / total_weight

            results["final_score"] += weight * score

        results = results.sort_values(
            "final_score",
            ascending=False,
        ).reset_index(drop=True)

        results["rank"] = np.arange(len(results)) + 1

        results["query"] = query

        results["parsed_fashion"] = str(clause_texts)

        results["parsed_style"] = parsed["style"]

        results["parsed_context"] = parsed["context"]

        return results.head(top_k).copy()
