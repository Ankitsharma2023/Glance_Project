from pathlib import Path
import random
import shutil
import csv

SOURCE_DIR = Path(
    r"C:\Users\aayus\Downloads\val_test2020\test"
)

DEST_DIR = Path("data/raw")

MANIFEST_PATH = Path("data/metadata/manifest.csv")

NUM_IMAGES = 1000

RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    images = [
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    print(f"Found {len(images)} images")

    if len(images) < NUM_IMAGES:
        raise ValueError(
            f"Dataset contains only {len(images)} images. "
            f"Need at least {NUM_IMAGES}."
        )

    images = sorted(images)

    selected_images = random.sample(
        images,
        NUM_IMAGES,
    )

    manifest_rows = []

    for index, source_path in enumerate(selected_images):
        image_id = f"img_{index:04d}"

        destination_name = (
            f"{image_id}{source_path.suffix.lower()}"
        )

        destination_path = (
            DEST_DIR / destination_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        manifest_rows.append(
            {
                "image_id": image_id,
                "image_path": str(destination_path),
                "source": "fashionpedia_test",
                "original_filename": source_path.name,
            }
        )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "image_path",
                "source",
                "original_filename",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"Copied {len(selected_images)} images "
        f"to {DEST_DIR}"
    )

    print(
        f"Manifest saved to {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()