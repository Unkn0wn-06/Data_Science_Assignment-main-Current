"""Rebuild the canonical processed dataset from the untouched raw CSV."""

if __package__:
    from scripts import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

from src.cleaning.pipeline import PROCESSED_DATA_PATH, build_production_dataset


def main() -> None:
    """Run cleaning and print dimensions, output path, and validation result."""
    raw, prepared, validation = build_production_dataset()
    print(f"Raw shape: {raw.shape}")
    print(f"Final shape: {prepared.shape}")
    print(f"Output path: {PROCESSED_DATA_PATH}")
    print(f"Validation passed: {validation['prepared']['valid']}")
    print(f"Validation details: {validation['prepared']}")


if __name__ == "__main__":
    main()
