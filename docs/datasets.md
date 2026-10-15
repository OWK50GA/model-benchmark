# Dataset preparation and annotation contract

The schema version is `1.0`. Use `vision-lab schema dataset` to inspect the full
machine-readable contract. Unknown fields are rejected to catch misspellings.

## Preparing real data

1. Put local images under `datasets/images/`, preferably organized by capture
   session or garment. Record paths relative to this root, with `/` separators.
2. Record the dimensions after applying EXIF orientation. Do not resize an image
   after annotating it without updating dimensions and boxes.
3. Assign stable `image_id` values and globally unique `annotation_id` values.
   Add one garment record per visible garment, including multiple garments in an
   outfit. Set `garments_complete=true` only after checking the whole image.
4. Use a documented vocabulary consistently for categories, subcategories,
   colors, patterns, silhouette, and visible details. These are nonempty strings,
   not a fixed fashion taxonomy; new labels do not require application changes.
5. Assign `identity_id` from collection/capture provenance, not from visual
   similarity. Reuse it for photographs of the same physical item. Use a different
   ID for separate copies of the same product. Leave it null if uncertain.
6. Group similar-looking distinct garments using `appearance_group`. Shared
   groups are candidate hard negatives, not identity labels.
7. Have a person review labels. Set `verification=verified` and `verified_by` only
   after review. Use `unverified` while work is incomplete. This records a claim
   of review; software cannot prove that the annotations are correct.
8. Validate the annotation file and then validate with `--check-images`.

The example has three fictional images and four garment observations: a shirt
seen from front and back, trousers worn with that shirt, and a different similar
shirt. No corresponding images or real annotations are supplied. It is marked
`synthetic=true`, and all garments have `verification=example`. Synthetic data
cannot contain `verified` records, and real datasets cannot contain `example`
records. Structural validity never makes an example eligible for real scoring.

## Coordinates and missing information

Boxes use continuous pixel coordinates in the EXIF-oriented full image:
`x_min`, `y_min`, `x_max`, `y_max`; origin is top-left and maxima are exclusive.
The full box for a 640×800 image is `(0, 0, 640, 800)`. Boxes must have positive
area and remain within image dimensions. Fractional coordinates are allowed.
The lab does not silently clip, normalize, or round boxes.

Boxes are optional. Category, identity, and individual attributes may be null.
`null` means unknown/unannotated; a list `[]` means a reviewed absence of values.
For example, `distinctive_details=null` is not the same as
`distinctive_details=[]`. A solid surface can be labeled `patterns=["solid"]`
under the chosen vocabulary. Describe apparent texture or sheen under
`visible_material_characteristics`; do not infer exact fiber composition from a
photograph. Use garment notes for limits of a partial annotation. Future metrics
must mask unknown attributes and report eligible denominators explicitly.

`garments=[]` with `garments_complete=true` denotes an annotated negative image;
with `false` it denotes an incomplete image. Detection benchmarks must not treat
incomplete annotations as exhaustive. The completeness flag is annotator-supplied,
so review negative images too.

Image contexts: `product`, `selfie`, `flat_lay`, `full_outfit`, `other`, `unknown`.
Viewpoints are garment-specific: `front`, `back`, `side`, `detail`, `unknown`.
The same photograph may show different viewpoints of different garments.

## JSON and JSONL

JSON stores one object with `schema_version`, `dataset_id`, `description`,
`synthetic`, and `images`. See the committed example.

JSONL stores one dataset metadata object on the first nonempty line, without
`images`, followed by one complete image object per line. Blank lines are ignored.
This preserves provenance and supports the same validation as JSON. The current
loader reads the whole dataset into memory; JSONL is not yet a streaming runner.

Convert explicitly through the public helpers (destination must not exist):

```python
from pathlib import Path
from fashionista_vision_lab.datasets import load_dataset, save_dataset

dataset = load_dataset(Path("datasets/annotations/example.json"))
save_dataset(dataset, Path("datasets/annotations/example-copy.jsonl"))
```

File checks reject absolute/traversing annotation paths, symlinks escaping the
image root, unreadable files, corrupt images, and mismatched dimensions. They do
not check whether the pictured garment agrees with the labels or detect duplicate
image content under different paths. Validation/inspection never modifies images.

## Designing later evaluations

Collect product photos, flat lays, worn views, selfies, lighting/background
changes, and difficult negatives. Include multiple viewpoints of each known
identity. Keep the same physical identity in one train/validation/test split;
inspection warns about identities crossing explicit splits. Within an evaluation
split, a query must have a separate positive gallery photograph. Exclude the
query itself and near-duplicate captures from trivial retrieval matches.

Identity IDs are dataset-scoped. Cross-dataset comparisons require an explicit ID
mapping. Avoid tuning matching thresholds on the test set. Record uncertain
identity separately rather than treating unlabeled garments as negatives. Phase 3
implements explicit query/gallery and verified-data eligibility rules; see
[identity evaluation](phase-3.md). Threshold selection remains your validation-set
decision. The dataset schema alone does not establish benchmark eligibility.
