"""
push_model_to_hf.py — Upload a saved model folder to the Hugging Face Hub.

Trained artifacts are written to ``outputs/trained_models/<exp>_<model>_<condition>_seed<seed>/``
by the cross-comparison runner (only the first seed is kept by default). This script
uploads one such folder to a Hugging Face model repository.

The token is read from the ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` environment variable, or
prompted securely via getpass — it is never hardcoded or logged.

Examples
--------
Upload a specific folder::

    python push_model_to_hf.py \
        --path outputs/trained_models/exp06_svm_router_BEREL_3.0_exp07_before_exp01_baseline_seed42 \
        --repo-id your-username/berel-svm-fusion

Auto-locate a BEREL SVM router folder (newest match) and upload::

    python push_model_to_hf.py --find-svm --model berel --repo-id your-username/berel-svm-fusion
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "outputs" / "trained_models"

# Default Hugging Face account owner. A --repo-id without a "/" is prefixed with this.
DEFAULT_HF_OWNER = "msperka"

# Substrings that identify a model family inside a folder name.
_MODEL_ALIASES = {
    "berel": ["berel"],
    "dictabert": ["dictabert"],
    "hero": ["hero"],
    "alephbertgimmel": ["alephbertgimmel"],
}


def _resolve_token() -> str:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
    if not token:
        # getpass keeps the token off the screen and out of shell history / logs.
        token = getpass.getpass("Hugging Face write token (input hidden): ").strip()
    if not token:
        raise SystemExit("No Hugging Face token provided. Aborting.")
    return token


def _find_folder(args: argparse.Namespace) -> Path:
    if args.path:
        p = Path(args.path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.is_dir():
            raise SystemExit(f"Model folder not found: {p}")
        return p

    if not TRAINED_MODELS_DIR.is_dir():
        raise SystemExit(f"No trained_models directory at {TRAINED_MODELS_DIR}")

    needles: list[str] = []
    if args.find_svm:
        needles.append("svm_router")
    if args.model:
        aliases = _MODEL_ALIASES.get(args.model.lower(), [args.model.lower()])
        needles.append(aliases)  # type: ignore[arg-type]
    if args.condition:
        needles.append(args.condition.lower())
    if args.seed:
        needles.append(f"seed{args.seed}".lower())

    def _matches(name: str) -> bool:
        low = name.lower()
        for needle in needles:
            if isinstance(needle, list):
                if not any(alias in low for alias in needle):
                    return False
            elif needle not in low:
                return False
        return True

    candidates = sorted(
        (d for d in TRAINED_MODELS_DIR.iterdir() if d.is_dir() and _matches(d.name)),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "No matching model folder found under outputs/trained_models. "
            f"Filters: {[c for c in needles]}"
        )
    if len(candidates) > 1:
        print("[info] Multiple matches; uploading the most recently modified:")
        for c in candidates:
            print(f"    {c.name}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a saved model folder to the Hugging Face Hub.")
    parser.add_argument("--path", help="Explicit path to the model folder to upload.")
    parser.add_argument("--find-svm", action="store_true", help="Auto-locate an SVM router folder.")
    parser.add_argument("--model", help="Model family filter for auto-locate (berel/dictabert/hero/alephbertgimmel).")
    parser.add_argument("--condition", help="Condition-key substring filter for auto-locate.")
    parser.add_argument("--seed", help="Seed filter for auto-locate (e.g. 42).")
    parser.add_argument("--repo-id", required=True, help="Target HF repo. A bare name is prefixed with '%s/', e.g. 'berel-svm-fusion' -> '%s/berel-svm-fusion'." % (DEFAULT_HF_OWNER, DEFAULT_HF_OWNER))
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--commit-message", default="Upload thesis NER model", help="Commit message.")
    args = parser.parse_args()

    repo_id = args.repo_id if "/" in args.repo_id else f"{DEFAULT_HF_OWNER}/{args.repo_id}"

    folder = _find_folder(args)
    print(f"[upload] Folder : {folder}")
    print(f"[upload] Repo   : {repo_id} ({'private' if args.private else 'public'})")

    from huggingface_hub import HfApi

    token = _resolve_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    url = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=args.commit_message,
    )
    print(f"[done] Uploaded to: https://huggingface.co/{repo_id}")
    print(f"[done] Commit: {url}")


if __name__ == "__main__":
    main()
