# Multimodal Fashion & Context Retrieval

An intelligent multimodal image retrieval system for searching fashion images using natural-language descriptions that combine **garment attributes, colours, style, and environmental context**.

The project was developed for a fashion retrieval task focused on queries such as:

- `A person in a bright yellow raincoat`
- `Someone wearing a blue shirt sitting on a park bench`
- `Casual weekend outfit for a city walk`
- `Professional business attire inside a modern office`
- `A red tie and a white shirt in a formal setting`

Instead of treating the complete query as a single global concept, the system decomposes the query and independently scores **fashion, garment-level evidence, style, and scene context** before performing query-aware late fusion.

---

## 1. Problem Statement

Traditional CLIP-style retrieval systems encode an entire image and an entire text query into one embedding vector and rank images using cosine similarity.

This works well for broad concepts such as:

> `a red outfit`

However, global embeddings can become unreliable for compositional fashion queries such as:

> `a red tie and a white shirt`

A global image representation may strongly encode the concepts `red`, `white`, `shirt`, and `tie` without correctly preserving which colour belongs to which garment.

This can result in retrieval of:

> a red shirt with a white accessory

instead of:

> a red tie with a white shirt

The problem becomes harder when fashion constraints are combined with scene semantics:

> `professional business attire inside a modern office`

A fashion-specialized encoder is useful for garments and outfit style, but environmental concepts such as offices, parks, homes, and urban streets require stronger general scene understanding.

This project addresses these limitations using a **two-stage, multi-signal retrieval architecture**.

---

## 2. Core Idea

The proposed retriever separates the search problem into four semantic signals:

1. **Global fashion similarity** — overall fashion relevance.
2. **Garment-region similarity** — clause-to-garment attribute binding.
3. **Style similarity** — formal, casual, professional, weekend, and related style concepts.
4. **Context similarity** — office, park, street, city, and home environments.

The final ranking is produced through query-aware late fusion.

```text
Natural-Language Query
          |
          v
Structured Query Decomposition
          |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
 Fashion Clauses          Style Signal       Context Signal
          |                   |                   |
          v                   v                   v
 FashionCLIP Regions      FashionCLIP         Global CLIP
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                    Query-Aware Late Fusion
                              |
                              v
                         Top-K Images
```

---

## 3. Dataset

The system uses a **1,000-image annotated Fashionpedia subset** prepared from locally available Fashionpedia imagery and Fashionpedia instance-attribute annotations.

Dataset preparation produced:

| Artifact | Count |
| --- | ---: |
| Images | 1,000 |
| Annotations in source split | 8,781 |
| Fashionpedia categories | 46 |
| Garment/apparel regions in selected corpus | 7,528 |
| Major fashion regions indexed | 4,027 |
| Region embedding dimension | 512 |

The 1,000-image corpus is intentionally preserved rather than reducing the dataset for implementation convenience.

### Why Fashionpedia?

Fashionpedia provides fine-grained apparel categories and garment-level annotations.

These annotations allow the retriever to move beyond global image similarity and compare individual text clauses against relevant garment regions.

Examples of indexed categories include:

```text
shirt / blouse
top / t-shirt / sweatshirt
sweater
cardigan
jacket
vest
pants
shorts
skirt
coat
dress
jumpsuit
tie
shoe
bag
scarf
hat
belt
```

### Dataset Limitation

Fashionpedia is primarily apparel-centric.

It provides strong garment supervision but does not guarantee balanced environmental coverage across:

```text
office
park
home
urban street
```

A pure CLIP context audit for:

```text
inside a modern office
```

showed weak office-scene coverage in the selected corpus.

Therefore, context-heavy retrieval is partly bounded by corpus coverage.

This limitation is documented rather than hidden through manual result replacement or fabricated metrics.

---

## 4. System Architecture

### 4.1 Global CLIP Index

A general CLIP encoder generates normalized image embeddings for the complete image.

Generated artifacts:

```text
data/features/global_clip.npy
data/features/global_clip.faiss
```

The global CLIP index is primarily used for scene and environmental semantics.

Examples include:

```text
inside an office
green park
urban street
city walk
home interior
```

---

### 4.2 Global FashionCLIP Index

FashionCLIP is used as the fashion-domain encoder.

The model used is:

```text
patrickjohncyh/fashion-clip
```

FashionCLIP is applied to complete images to create a fashion-oriented global index.

Generated artifacts:

```text
data/features/global_fashionclip.npy
data/features/global_fashionclip.faiss
```

This representation is used for garment, outfit, and style semantics.

---

### 4.3 Garment Region Extraction

Fashionpedia bounding-box annotations are used to identify garment regions.

The dataset preparation pipeline creates:

```text
data/metadata/garment_regions.csv
```

Each region is associated with information such as:

```text
image ID
Fashionpedia category ID
category name
bounding-box coordinates
```

Only major fashion categories are included in the main region retrieval index.

Fine structural annotations such as:

```text
neckline
sleeve
zipper
pocket
rivet
```

are excluded from the primary garment retrieval index to reduce irrelevant region matches.

The final region index contains **4,027 major fashion regions**.

---

### 4.4 Region FashionCLIP Embeddings

Each garment crop is independently encoded using FashionCLIP.

The resulting normalized 512-dimensional embeddings are stored in:

```text
data/features/region_fashionclip.npy
data/features/region_fashionclip.faiss
data/features/region_mapping.csv
```

This creates a multi-vector image representation.

Instead of representing an image only as:

```text
Image -> Global Vector
```

the system additionally represents it as:

```text
Image
 |
 +-- Shirt Region Vector
 +-- Jacket Region Vector
 +-- Pants Region Vector
 +-- Tie Region Vector
 +-- Dress Region Vector
 +-- ...
```

This region-level representation is used during second-stage reranking.

---

## 5. Structured Query Decomposition

The retriever decomposes a natural-language query into:

```text
fashion clauses
style
context
```

For example:

```text
A red tie and a white shirt in a formal setting
```

is parsed as:

```text
Fashion clauses: ["red tie", "white shirt"]
Style: formal
Context: ""
```

Another query:

```text
Professional business attire inside a modern office
```

is parsed as:

```text
Fashion clauses: []
Style: professional business
Context: inside office
```

The current implementation uses a lightweight deterministic parser based on:

```text
colour vocabulary
garment vocabulary
style vocabulary
context vocabulary
```

This approach was selected because it is:

```text
reproducible
fast
dependency-light
easy to inspect
sufficient for the assignment query domain
```

A structured LLM-based query parser is discussed as future work for more complex natural-language phrasing.

---

## 6. Category-Constrained Clause-to-Region MaxSim

This is the primary compositional retrieval component.

Consider the clause:

```text
red tie
```

The system does **not** compare this clause against every region in an image.

The parser identifies:

```text
garment = tie
Fashionpedia category ID = 16
```

Only tie-compatible regions are considered.

For text clause `q_i` and compatible image regions `r_j`, the clause score is:

```text
score(q_i, image) = max_j similarity(q_i, r_j)
```

This is a clause-to-region MaxSim operation inspired by late-interaction retrieval.

For:

```text
red tie + white shirt
```

the system independently calculates:

```text
red tie
    |
    v
best compatible tie region

white shirt
    |
    v
best compatible shirt/top region
```

This reduces concept leakage where the colour `red` could otherwise match a skirt, dress, or background even though the query explicitly binds it to `tie`.

---

## 7. Conjunctive Region Scoring

MaxSim alone can still produce a high score when only one query clause is strongly satisfied.

For a query with clause scores:

```text
[
    red tie score,
    white shirt score
]
```

the system computes both the mean and minimum clause score.

For multi-clause queries:

```text
region_score =
    0.60 * mean(clause_scores)
    + 0.40 * min(clause_scores)
```

The minimum term penalizes an image that strongly satisfies one clause while ignoring another.

For single-clause queries:

```text
region_score = clause_score
```

---

## 8. Coverage Scoring

A clause is considered matched when:

```text
clause_score >= 0.25
```

Coverage is calculated as:

```text
matched clauses / total clauses
```

Example:

```text
red tie     -> matched
white shirt -> not matched

coverage = 1 / 2 = 0.50
```

This gives the retriever an explicit measure of how much of the query was satisfied.

### All-Clauses-Matched Signal

For multi-clause queries, an additional binary conjunction signal is used:

```text
1.0 -> every clause passed the threshold
0.0 -> at least one clause failed
```

This explicitly rewards complete compositional matches.

---

## 9. Context-Aware Retrieval

FashionCLIP is specialized for fashion semantics.

Environmental context is therefore scored separately using general CLIP embeddings of the complete image.

A garment crop cannot reliably determine whether a person is:

```text
inside an office
in a park
at home
walking through a city
```

Context is therefore evaluated using **global image embeddings**.

---

## 10. Contrastive Context Prototypes

Raw context similarity can be weakly discriminative.

For example, a generic indoor image may receive moderate office similarity even when the image actually contains a closet or dressing room.

The system defines positive and negative context prototypes.

For the `office` context:

```text
Positive prototypes:
- inside a modern office
- professional office interior
- corporate workplace
- business office environment
- indoor office workspace
```

Negative prototypes:

```text
- outdoor street
- urban sidewalk
- fashion runway
- park
- home interior
- closet or dressing room
```

The context score is calculated as:

```text
context_score =
    positive_similarity
    - 0.50 * negative_similarity
```

This contrastive formulation attempts to reward scene evidence while suppressing related but incorrect environments.

Context prototype groups are currently implemented for:

```text
office
park
city
street
home
```

---

## 11. Dual Candidate Generation

A reranker cannot recover an image that was removed during first-stage retrieval.

Using only FashionCLIP candidates creates a candidate-recall problem for context-heavy queries.

The system therefore retrieves:

```text
Top 100 FashionCLIP candidates
+
Top 100 CLIP candidates
```

The candidate IDs are unioned and deduplicated.

```text
FashionCLIP Top-100 ----+
                        |
                        +--> Union --> Advanced Reranker
                        |
CLIP Top-100 -----------+
```

FashionCLIP contributes fashion-oriented candidates.

CLIP contributes scene and context-oriented candidates.

For one tested office query, the candidate diagnostic was:

```text
FashionCLIP candidates = 100
CLIP candidates        = 100
Union candidates       = 125
```

The context index therefore introduced additional candidates that were absent from the FashionCLIP top 100.

---

## 12. Style Scoring

Style concepts are independently encoded using FashionCLIP.

Examples include:

```text
professional
business
formal
casual
weekend
smart
relaxed
elegant
sporty
streetwear
```

For a parsed style query, the style text embedding is compared with the global FashionCLIP image embedding.

Example:

```text
Casual weekend outfit for a city walk
```

produces:

```text
Style: casual weekend
Context: city walk
```

The style and context signals are scored independently before fusion.

---

## 13. Query-Aware Late Fusion

CLIP and FashionCLIP are separate learned representation spaces.

The system does not concatenate their embeddings and assume that the resulting dimensions share a common semantic geometry.

Instead, each semantic signal is independently scored and combined during ranking.

Available components are:

```text
global fashion similarity
region MaxSim
coverage
all-clause conjunction
style similarity
context similarity
```

Base component weights are:

| Component | Weight |
| --- | ---: |
| Global fashion | 0.25 |
| Region match | 0.30 |
| Coverage | 0.15 |
| Conjunction | 0.10 |
| Style | 0.15 |
| Context | 0.15 |

Only components relevant to the parsed query are activated.

The active weights are normalized before final scoring.

For example:

```text
yellow raincoat
```

does not force a context signal.

A query such as:

```text
casual weekend outfit for a city walk
```

activates:

```text
global fashion
style
context
```

This produces query-aware late fusion.

---

## 14. Complete Retrieval Pipeline

The full retrieval flow is:

```text
1. Receive natural-language query

2. Parse:
   - fashion clauses
   - style
   - context

3. Encode full query with FashionCLIP

4. Encode full query with CLIP

5. Retrieve FashionCLIP top-100 candidates

6. Retrieve CLIP top-100 candidates

7. Union and deduplicate candidates

8. Encode individual fashion clauses

9. Restrict each clause to compatible garment categories

10. Compute clause-to-region MaxSim

11. Compute conjunctive region score

12. Compute coverage

13. Compute all-clause match signal

14. Compute FashionCLIP style score

15. Compute contrastive CLIP context score

16. Min-max normalize active ranking components

17. Perform query-aware late fusion

18. Sort candidates by final score

19. Return top-k images
```

---

## 15. Qualitative Results

### Query 1: Bright Yellow Raincoat

```text
A person in a bright yellow raincoat
```

An observed top result produced:

```text
global score   = 0.365
region score   = 0.317
coverage       = 1.00
all matched    = 1
```

The region module improved garment specificity compared with early global retrieval, which frequently returned broadly yellow outfits.

---

### Query 2: Blue Shirt on a Park Bench

```text
Someone wearing a blue shirt sitting on a park bench
```

Under pure CLIP, the top-ranked candidate matched the seated outdoor context but violated the blue-shirt constraint.

The full retriever promoted a candidate containing:

```text
blue upper-body clothing
seated person
bench
outdoor greenery
```

to Rank 1.

This provides a qualitative example of improved attribute-context binding.

---

### Query 3: Casual Weekend City Walk

```text
Casual weekend outfit for a city walk
```

The top-ranked results contained casual outfits in:

```text
sidewalk scenes
street environments
urban walking environments
```

This query showed strong combined style-context retrieval.

---

### Query 4: Professional Business Attire in an Office

```text
Professional business attire inside a modern office
```

The system retrieved strong business-fashion candidates but weak office scenes.

A separate pure CLIP top-20 context audit for:

```text
inside a modern office
```

did not reveal a convincing modern corporate office image among the strongest retrieved candidates.

This indicates a corpus scene-coverage limitation.

The system does not claim that context scoring can retrieve an environment absent or poorly represented in the corpus.

---

### Query 5: Red Tie and White Shirt

```text
A red tie and a white shirt in a formal setting
```

The available annotated corpus contains very few tie regions.

The category-constrained MaxSim logic correctly restricts:

```text
red tie
```

to compatible tie regions.

However, exact retrieval quality remains limited by long-tail category coverage.

This demonstrates an important distinction between:

```text
ranking architecture limitation
```

and:

```text
corpus coverage limitation
```

---

## 16. Evaluation

The project includes a lightweight human-judged evaluation workflow.

The query benchmark is stored in:

```text
evaluation/queries.csv
```

The current evaluation set contains 15 queries across:

```text
garment retrieval
colour-garment retrieval
compositional retrieval
style retrieval
context retrieval
style-context retrieval
```

Run evaluation retrieval:

```bash
python evaluation/run_evaluation.py
```

The script retrieves the top five results for every evaluation query and writes:

```text
evaluation/retrieval_results.csv
```

### Human Relevance Judgments

Run:

```bash
python evaluation/label_results.py
```

Results can be graded as:

```text
2 = highly relevant
1 = partially relevant
0 = irrelevant
```

The evaluation framework is designed to support:

```text
Precision@5
Mean Reciprocal Rank
nDCG@5
```

after human relevance judgments are completed.

**No metric values are reported for incomplete judgments.**

The project intentionally avoids fabricated quantitative results.

---

## 17. Ablation Framework

The project includes a ranking-component ablation script:

```bash
python evaluation/run_ablation.py
```

The evaluated ranking configurations are:

```text
global_only

global_region

region_coverage

full_model
```

These configurations isolate the contribution of:

```text
global FashionCLIP similarity
garment-region evidence
coverage and conjunction
style and context late fusion
```

Generated ablation rankings are written to:

```text
evaluation/ablation_results.csv
```

### Ablation Scope

The current implementation performs a **shared-candidate-pool ranking ablation**.

It does not claim that each configuration is a separately rebuilt end-to-end retrieval architecture.

This distinction is important when interpreting the ablation results.

---

## 18. Streamlit Demo

The repository includes an interactive Streamlit application.

The interface is designed as a fashion-first visual discovery experience rather than a traditional machine-learning dashboard.

Features include:

```text
natural-language fashion search
parsed fashion/style/context intent
top-5 visual retrieval results
ranking scores
expandable result explanations
global fashion evidence
region evidence
coverage
style score
context score
```

Run:

```bash
streamlit run app.py
```

Example query:

```text
Someone wearing a blue shirt sitting on a park bench
```

The interface displays:

```text
understood fashion constraints
understood style
understood context
top five visual matches
retrieval explanation for each result
```

---

## 19. Project Structure

```text
multimodal-fashion-retrieval/
|
|-- app.py
|-- requirements.txt
|-- requirements-lock.txt
|-- README.md
|
|-- data/
|   |
|   |-- features/
|   |   |-- image_mapping.csv
|   |   `-- region_mapping.csv
|   |
|   `-- metadata/
|       |-- garment_regions.csv
|       |-- instances_attributes_val2020.json
|       `-- manifest.csv
|
|-- evaluation/
|   |-- queries.csv
|   |-- run_evaluation.py
|   |-- label_results.py
|   `-- run_ablation.py
|
|-- scripts/
|   |-- prepare_dataset.py
|   |-- prepare_annotated_dataset.py
|   |-- audit_dataset.py
|   |-- visualize_audit.py
|   |-- build_index.py
|   |-- build_fashion_index.py
|   |-- build_region_index.py
|   |-- compare_models.py
|   |-- search.py
|   |-- visual_search.py
|   `-- advanced_search.py
|
`-- src/
    |
    |-- indexer/
    |   `-- encoders.py
    |
    `-- retriever/
        `-- fashion_context_retriever.py
```

---

## 20. Installation

### Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd multimodal-fashion-retrieval
```

### Create a Virtual Environment

Windows Command Prompt:

```bash
python -m venv venv
venv\Scripts\activate
```

Git Bash on Windows:

```bash
python -m venv venv
source venv/Scripts/activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Pretrained encoder weights are downloaded from Hugging Face when the models are loaded for the first time.

An authenticated Hugging Face token is optional but can provide higher Hub rate limits.

---

## 21. Data Preparation

Raw images and generated embedding indexes are intentionally excluded from Git because of repository size.

The project uses Fashionpedia annotations and locally available Fashionpedia images.

Run:

```bash
python scripts/prepare_annotated_dataset.py
```

The preparation pipeline:

```text
loads Fashionpedia annotations
finds locally available annotated images
selects 1,000 annotated images
backs up the previous baseline corpus
copies selected images
creates the image manifest
creates garment-region metadata
```

Expected prepared artifacts include:

```text
data/raw/
data/metadata/manifest.csv
data/metadata/garment_regions.csv
```

---

## 22. Building the Indexes

### Build Global CLIP Index

```bash
python scripts/build_index.py
```

Generated artifacts:

```text
data/features/global_clip.npy
data/features/global_clip.faiss
data/features/image_mapping.csv
```

### Build Global FashionCLIP Index

```bash
python scripts/build_fashion_index.py
```

Generated artifacts:

```text
data/features/global_fashionclip.npy
data/features/global_fashionclip.faiss
```

### Build Region FashionCLIP Index

```bash
python scripts/build_region_index.py
```

Generated artifacts:

```text
data/features/region_fashionclip.npy
data/features/region_fashionclip.faiss
data/features/region_mapping.csv
```

The generated `.npy` and `.faiss` files are excluded from Git and must be rebuilt locally.

---

## 23. Running Search

### Basic CLIP Search

```bash
python scripts/search.py
```

### Advanced Multimodal Retrieval

```bash
python scripts/advanced_search.py
```

Example:

```text
Enter search query:
A red tie and a white shirt in a formal setting
```

The advanced retrieval script displays:

```text
parsed fashion clauses
parsed style
parsed context
candidate-generation diagnostics
final score
global score
region score
coverage
all-clause match
style score
context score
clause-level scores
```

---

## 24. Programmatic Retrieval

The reusable retrieval engine can be imported directly:

```python
from src.retriever.fashion_context_retriever import (
    FashionContextRetriever,
)

retriever = FashionContextRetriever()

results = retriever.search(
    query=(
        "Someone wearing a blue shirt "
        "sitting on a park bench"
    ),
    top_k=5,
)

print(
    results[
        [
            "rank",
            "image_id",
            "final_score",
        ]
    ]
)
```

The same retrieval engine powers:

```text
CLI search
evaluation
Streamlit application
```

This prevents duplication of ranking logic across interfaces.

---

## 25. Scalability Toward One Million Images

The current prototype contains 1,000 images, but the architecture is based on a two-stage retrieval strategy.

A naive exhaustive region comparison over one million images would be computationally expensive.

The proposed scaling strategy is:

```text
Stage 1
Approximate Global Retrieval
            |
            v
      Top-N Candidates
            |
            v
Stage 2
Region-Aware Reranking
```

At larger scale:

1. Use FAISS IVF, HNSW, or product-quantized indexes for global candidate retrieval.
2. Shard global indexes across machines or semantic partitions.
3. Store region vectors separately from global image vectors.
4. Fetch region embeddings only for first-stage survivors.
5. Batch text and region similarity operations on GPU.
6. Cache frequent query embeddings and parsed query structures.
7. Apply metadata filters before dense retrieval where appropriate.
8. Consider compressed region representations or a learned late-interaction index.

The key scalability property is that region MaxSim is **not executed against every garment region in the entire corpus**.

It is applied only to first-stage candidate survivors.

---

## 26. Extension to Cities and Places

The architecture naturally supports additional semantic signals.

A location or place signal could be added as:

```text
fashion
style
context
location
```

Example queries:

```text
casual outfit near a beach

winter streetwear in New York

formal outfit in a hotel lobby

summer clothing in a European city
```

Potential location signals include:

```text
visual place embeddings
landmark recognition
geospatial metadata
city tags
place classifiers
```

The location score could be introduced as another late-fusion component without forcing it into the FashionCLIP representation space.

---

## 27. Weather-Aware Retrieval

Weather can be introduced as another independently scored semantic axis.

Examples:

```text
sunny
rainy
snowy
cold
humid
windy
```

A production system could combine:

```text
visual weather evidence
garment suitability
live weather metadata
location
season
```

For example:

```text
outfit for a rainy city walk
```

could retrieve waterproof outerwear while also requiring rainy or wet-street context.

---

## 28. Future Precision Improvements

### Structured LLM Query Parsing

The deterministic parser can be replaced with constrained structured output.

Example:

```json
{
  "fashion_clauses": [
    {
      "color": "red",
      "garment": "tie"
    },
    {
      "color": "white",
      "garment": "shirt"
    }
  ],
  "style": "formal",
  "context": "office"
}
```

This would improve robustness to complex phrasing.

### Explicit Colour Modeling

A dedicated colour module could extract garment-region colour in LAB colour space.

This would explicitly separate:

```text
garment identity
```

from:

```text
garment colour
```

and reduce dependence on vision-language embedding colour sensitivity.

### Context-Balanced Corpus

A coverage-aware dataset selection pipeline could explicitly enforce minimum scene counts for:

```text
office
urban street
park
home
```

The current office audit shows why this matters.

### Multi-Person Garment Association

For images containing multiple people, garment regions should be associated with person instances.

Without person association, the system could theoretically combine:

```text
red tie on person A
+
white shirt on person B
```

into one image-level compositional match.

### Learned Fusion

The current fusion weights are manually specified and query-aware.

With a larger relevance-labeled dataset, a learning-to-rank model could learn weights using:

```text
global score
region score
coverage
conjunction
style score
context score
```

### Encoder Comparison

Future experiments could compare:

```text
FashionCLIP
SigLIP
Marqo-FashionSigLIP
larger OpenCLIP variants
```

using the same relevance benchmark.

---

## 29. Key Design Decisions

| Decision | Motivation |
| --- | --- |
| FashionCLIP for fashion | Domain-specific garment and outfit semantics |
| CLIP for context | General scene representation |
| Separate scoring spaces | CLIP and FashionCLIP are different learned spaces |
| Garment regions | Improves attribute-garment binding |
| Category-constrained MaxSim | Prevents colour matching unrelated garments |
| Coverage | Rewards satisfying more clauses |
| Conjunction signal | Rewards complete multi-clause matches |
| Dual candidate generation | Protects context candidate recall |
| Late fusion | Combines heterogeneous semantic signals |
| Two-stage retrieval | Limits expensive region scoring |
| Human relevance workflow | Supports honest ranking evaluation |

---

## 30. Known Limitations

The current prototype has several known limitations:

- The selected Fashionpedia corpus has limited office-scene coverage.
- Tie regions are a long-tail category in the available annotations.
- Query parsing uses a fixed vocabulary and deterministic rules.
- Colour is not explicitly modeled using a dedicated colour space.
- Multi-person garment ownership is not resolved.
- Context prototypes are manually defined.
- Fusion weights are manually specified.
- Complete human relevance labeling has not been reported as a finished quantitative benchmark.
- Raw images and generated FAISS indexes are not committed to Git and must be prepared locally.

These limitations are treated as explicit engineering tradeoffs and future-work directions.

---

## 31. Summary

This project implements a multimodal fashion retrieval engine that goes beyond single-vector CLIP search.

The retrieval architecture combines:

```text
FashionCLIP global retrieval
+
CLIP context retrieval
+
dual candidate generation
+
Fashionpedia garment regions
+
category-constrained clause-to-region MaxSim
+
conjunctive scoring
+
coverage scoring
+
style scoring
+
contrastive context scoring
+
query-aware late fusion
```

The core motivation is to separately model:

> **what a person is wearing**

and:

> **where the person is and what style the outfit represents**

The prototype also exposes an important retrieval-system constraint:

> **A ranking model cannot retrieve a concept that is absent or poorly represented in its corpus.**

Strong multimodal retrieval therefore requires both semantic ranking logic and dataset coverage across the intended search axes.

---

## Author

**Aayushi Gupta**

Multimodal Fashion & Context Retrieval prototype.
