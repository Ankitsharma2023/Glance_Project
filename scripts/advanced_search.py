from pathlib import Path
import sys
import re
import ast

import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.indexer.encoders import (
    CLIPEncoder,
    FashionCLIPEncoder,
)

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "advanced"

CANDIDATE_K = 100
TOP_K = 5

# -------------------------------------------------------
# Fusion weights
# -------------------------------------------------------

W_GLOBAL = 0.30
W_REGION = 0.35
W_COVERAGE = 0.20
W_CONTEXT = 0.15

COVERAGE_THRESHOLD = 0.25


# -------------------------------------------------------
# Query vocabulary
# -------------------------------------------------------

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


# -------------------------------------------------------
# Query decomposition
# -------------------------------------------------------


def parse_query(query):
    text = query.lower()

    normalized = re.sub(
        r"[^a-z0-9\- ]+",
        " ",
        text,
    )

    tokens = normalized.split()

    fashion_clauses = []
    used_positions = set()

    # Color-garment bindings.
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

            used_positions.add(index)
            used_positions.add(index + 1)

    # Garment-only clauses not already captured.
    captured_garments = {clause["garment"] for clause in fashion_clauses}

    for index, token in enumerate(tokens):
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

        used_positions.add(index)

    style_tokens = [token for token in tokens if token in STYLE_TERMS]

    context_tokens = [token for token in tokens if token in CONTEXT_TERMS]

    return {
        "fashion_clauses": fashion_clauses,
        "style": " ".join(style_tokens),
        "context": " ".join(context_tokens),
    }


# -------------------------------------------------------
# Contrastive context prototype bank
# -------------------------------------------------------

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

CONTEXT_NEGATIVE_WEIGHT = 0.50


def get_context_prototypes(context_text):
    tokens = set(context_text.lower().split())

    for context_name in ("office", "park", "city", "street", "home"):
        if context_name in tokens:
            return CONTEXT_PROTOTYPES[context_name]

    return None


# -------------------------------------------------------
# Score normalization
# -------------------------------------------------------


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


# -------------------------------------------------------
# Main retrieval
# -------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load global image mapping/indexes
    # --------------------------------------------------

    image_mapping = pd.read_csv(FEATURE_DIR / "image_mapping.csv")

    fashion_index = faiss.read_index(str(FEATURE_DIR / "global_fashionclip.faiss"))
    clip_index = faiss.read_index(str(FEATURE_DIR / "global_clip.faiss"))

    clip_embeddings = np.load(FEATURE_DIR / "global_clip.npy").astype(np.float32)

    fashion_embeddings = np.load(FEATURE_DIR / "global_fashionclip.npy").astype(
        np.float32
    )

    # --------------------------------------------------
    # Load region features
    # --------------------------------------------------

    region_embeddings = np.load(FEATURE_DIR / "region_fashionclip.npy").astype(
        np.float32
    )

    region_mapping = pd.read_csv(FEATURE_DIR / "region_mapping.csv")

    if len(region_embeddings) != len(region_mapping):
        raise RuntimeError("Region embedding/mapping mismatch.")

    # --------------------------------------------------
    # Encoders
    # --------------------------------------------------

    fashion_encoder = FashionCLIPEncoder()
    context_encoder = CLIPEncoder()

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = input("Enter search query: ").strip()

    parsed = parse_query(query)

    print("\nParsed query:")

    print(
        "Fashion clauses:",
        [clause["text"] for clause in parsed["fashion_clauses"]],
    )

    print(
        "Style:",
        parsed["style"],
    )

    print(
        "Context:",
        parsed["context"],
    )

    routed_context = get_context_prototypes(parsed["context"])
    print(
        "Context scoring:",
        (
            "contrastive prototypes"
            if routed_context is not None
            else "direct CLIP similarity"
        ),
    )

    # --------------------------------------------------
    # Stage 1: dual candidate generation
    # --------------------------------------------------

    fashion_query_embedding = fashion_encoder.encode_texts([query])
    clip_query_embedding = context_encoder.encode_texts([query])

    fashion_candidate_scores, fashion_candidate_indices = fashion_index.search(
        fashion_query_embedding,
        CANDIDATE_K,
    )
    clip_candidate_scores, clip_candidate_indices = clip_index.search(
        clip_query_embedding,
        CANDIDATE_K,
    )

    fashion_candidate_indices = fashion_candidate_indices[0]
    clip_candidate_indices = clip_candidate_indices[0]

    # Union preserves FashionCLIP candidates first, then adds unseen CLIP
    # context candidates. Stage 2 computes comparable FashionCLIP global
    # scores directly from the stored normalized embeddings.
    candidate_indices = np.asarray(
        list(
            dict.fromkeys(
                fashion_candidate_indices.tolist() + clip_candidate_indices.tolist()
            )
        ),
        dtype=np.int64,
    )

    global_scores = (
        fashion_embeddings[candidate_indices] @ fashion_query_embedding[0]
    ).astype(np.float32)

    print(
        "Candidate generation:",
        f"FashionCLIP={len(set(fashion_candidate_indices.tolist()))},",
        f"CLIP={len(set(clip_candidate_indices.tolist()))},",
        f"union={len(candidate_indices)}",
    )

    # --------------------------------------------------
    # Fashion clause embeddings
    # --------------------------------------------------

    fashion_clauses = parsed["fashion_clauses"]

    if fashion_clauses:
        clause_texts = [clause["text"] for clause in fashion_clauses]

        clause_embeddings = fashion_encoder.encode_texts(clause_texts)
    else:
        clause_texts = []
        clause_embeddings = None

    # --------------------------------------------------
    # Context embedding
    # --------------------------------------------------

    context_text = parsed["context"]
    context_prototypes = get_context_prototypes(context_text)

    if context_text:
        context_embedding = (context_encoder.encode_texts([context_text]))[0]
    else:
        context_embedding = None

    if context_prototypes is not None:
        positive_context_embeddings = context_encoder.encode_texts(
            context_prototypes["positive"]
        )
        negative_context_embeddings = context_encoder.encode_texts(
            context_prototypes["negative"]
        )
    else:
        positive_context_embeddings = None
        negative_context_embeddings = None

    style_text = parsed["style"]

    if style_text:
        style_embedding = (fashion_encoder.encode_texts([style_text]))[0]
    else:
        style_embedding = None

    # --------------------------------------------------
    # Region rows grouped by image
    # --------------------------------------------------

    region_groups = {
        image_id: group for image_id, group in region_mapping.groupby("image_id")
    }

    result_rows = []

    # --------------------------------------------------
    # Stage 2: constrained region reranking
    # --------------------------------------------------

    for (
        candidate_position,
        global_score,
    ) in zip(
        candidate_indices,
        global_scores,
    ):
        image_row = image_mapping.iloc[candidate_position]

        image_id = image_row["image_id"]

        clause_scores = []

        if fashion_clauses and image_id in region_groups:
            image_regions = region_groups[image_id]

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

                compatible_embeddings = region_embeddings[region_indices]

                similarities = compatible_embeddings @ clause_embeddings[clause_index]

                clause_scores.append(float(similarities.max()))

        elif fashion_clauses:
            clause_scores = [0.0 for _ in fashion_clauses]

        # ----------------------------------------------
        # Conjunctive MaxSim aggregation
        # ----------------------------------------------

        if clause_scores:
            clause_array = np.asarray(clause_scores, dtype=np.float32)
            mean_clause_score = float(np.mean(clause_array))
            min_clause_score = float(np.min(clause_array))

            if len(clause_array) > 1:
                region_score = 0.60 * mean_clause_score + 0.40 * min_clause_score
            else:
                region_score = mean_clause_score

            matched_clauses = clause_array >= COVERAGE_THRESHOLD
            coverage_score = float(np.mean(matched_clauses))
            all_clauses_matched = float(np.all(matched_clauses))
        else:
            region_score = 0.0
            coverage_score = 0.0
            all_clauses_matched = 0.0

        # ----------------------------------------------
        # Context score
        # ----------------------------------------------

        if context_embedding is not None:
            image_context_embedding = clip_embeddings[candidate_position]

            if positive_context_embeddings is not None:
                positive_score = float(
                    np.mean(positive_context_embeddings @ image_context_embedding)
                )

                negative_score = float(
                    np.max(negative_context_embeddings @ image_context_embedding)
                )

                context_score = (
                    positive_score - CONTEXT_NEGATIVE_WEIGHT * negative_score
                )
            else:
                context_score = float(image_context_embedding @ context_embedding)
        else:
            context_score = 0.0

        if style_embedding is not None:
            style_score = float(
                fashion_embeddings[candidate_position] @ style_embedding
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
                "all_clauses_matched": all_clauses_matched,
                "style_score": style_score,
                "context_score": context_score,
                "clause_scores": clause_scores,
            }
        )

    results = pd.DataFrame(result_rows)

    # --------------------------------------------------
    # Candidate-local normalization
    # --------------------------------------------------

    results["global_norm"] = minmax(results["global_score"].values)

    results["region_norm"] = minmax(results["region_score"].values)

    if style_embedding is not None:
        results["style_norm"] = minmax(results["style_score"].values)
    else:
        results["style_norm"] = 0.0

    if context_embedding is not None:
        results["context_norm"] = minmax(results["context_score"].values)
    else:
        results["context_norm"] = 0.0

    # --------------------------------------------------
    # Query-aware dynamic late fusion
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    safe_query = "".join(
        character if character.isalnum() else "_" for character in query
    )[:80]

    csv_path = OUTPUT_DIR / f"{safe_query}.csv"

    results.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    print("\nTop results:\n")

    for rank, result in results.head(TOP_K).iterrows():
        print(
            f"{rank + 1}. "
            f"{result['image_id']} | "
            f"final={result['final_score']:.3f} | "
            f"global={result['global_score']:.3f} | "
            f"region={result['region_score']:.3f} | "
            f"coverage={result['coverage_score']:.2f} | "
            f"all_matched={result['all_clauses_matched']:.0f} | "
            f"style={result['style_score']:.3f} | "
            f"context={result['context_score']:.3f}"
        )

        if clause_texts:
            print(
                "   Clause scores:",
                {
                    clause: round(
                        score,
                        3,
                    )
                    for clause, score in zip(
                        clause_texts,
                        result["clause_scores"],
                    )
                },
            )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    top_results = results.head(TOP_K)

    fig, axes = plt.subplots(
        1,
        TOP_K,
        figsize=(18, 5),
    )

    for rank, (
        axis,
        (_, result),
    ) in enumerate(
        zip(
            axes,
            top_results.iterrows(),
        ),
        start=1,
    ):
        image_path = PROJECT_ROOT / result["image_path"]

        image = Image.open(image_path).convert("RGB")

        axis.imshow(image)
        axis.axis("off")

        axis.set_title(
            f"Rank {rank}\n"
            f"Final: "
            f"{result['final_score']:.3f}\n"
            f"{result['image_id']}"
        )

    fig.suptitle(
        query,
        fontsize=18,
    )

    plt.tight_layout()

    output_image_path = OUTPUT_DIR / f"{safe_query}.png"

    plt.savefig(
        output_image_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"\nResults CSV: {csv_path}")

    print(
        "Visualization:",
        output_image_path,
    )

    plt.show()


if __name__ == "__main__":
    main()
