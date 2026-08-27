"""Build the saved Scenario B comparison for the four final assignment models."""

if __package__:
    from scripts import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

from src.models.final.final_evaluation import build_final_results


def main() -> None:
    result = build_final_results()
    print("\nFinal Scenario B comparison:")
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
