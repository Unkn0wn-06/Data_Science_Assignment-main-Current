"""Compatibility entry point for the active final Scenario B comparison."""

if __package__:
    from scripts import _bootstrap  # noqa: F401
else:
    import _bootstrap  # noqa: F401

from scripts.build_final_model_results import main


if __name__ == "__main__":
    main()
