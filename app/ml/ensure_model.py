"""Make sure a usable trained model exists; (re)train automatically if not.

Run: python -m app.ml.ensure_model
"""

from __future__ import annotations


def main() -> None:
    from app.ml.predict import _load_artifacts, reload_artifacts

    try:
        _load_artifacts()
        print("Model artifacts found and readable - skipping training.")
    except Exception as exc:  # noqa: BLE001 - missing OR incompatible pickle: retrain either way
        print(f"Model artifacts missing or unreadable ({type(exc).__name__}); training a new model...")
        from app.ml.train import train_and_select_best

        meta = train_and_select_best()
        reload_artifacts()
        print(f"Trained {meta['model_version']}.")


if __name__ == "__main__":
    main()
