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


def _find_folders(args: argparse.Namespace) -> list[Path]:
    if args.path:
        p = Path(args.path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.is_dir():
            raise SystemExit(f"Model folder not found: {p}")
        return [p]

    if not TRAINED_MODELS_DIR.is_dir():
        raise SystemExit(f"No trained_models directory at {TRAINED_MODELS_DIR}")

    needles: list = []
    if args.find_svm:
        needles.append("svm_router")
    if args.model:
        aliases = _MODEL_ALIASES.get(args.model.lower(), [args.model.lower()])
        needles.append(aliases)
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
            f"Filters: {needles}"
        )
    if len(candidates) > 1 and not args.all_conditions:
        listing = "\n".join(f"    {c.name}" for c in candidates)
        raise SystemExit(
            f"Multiple matching folders ({len(candidates)}):\n{listing}\n\n"
            "Narrow the selection with --condition <substring> (or --path <folder>), "
            "or pass --all-conditions to upload every match into per-condition subfolders."
        )
    return candidates


def _model_card(repo_id: str, folders: list[Path], is_svm: bool) -> str:
    """A minimal but valid HF model card with YAML metadata (fixes the empty-card warning)."""
    if is_svm:
        summary = (
            "scikit-learn `LinearSVC` disagreement router for NER fusion. It routes "
            "between a regular NER model (Exp01) and a cascaded pipeline (Exp04) on "
            "tokens where the two disagree. It is **not** a standalone transformer."
        )
        library = "sklearn"
        tags = ["ner", "hebrew", "fusion", "svm", "scikit-learn"]
    else:
        summary = "Hebrew NER model trained for the thesis experiments."
        library = "transformers"
        tags = ["ner", "hebrew", "token-classification"]

    listed = "\n".join(f"- `{f.name}`" for f in folders)
    return (
        "---\n"
        "license: mit\n"
        "language:\n- he\n"
        f"library_name: {library}\n"
        "tags:\n" + "".join(f"- {t}\n" for t in tags) +
        "---\n\n"
        f"# {repo_id.split('/')[-1]}\n\n"
        f"{summary}\n\n"
        "## Contents\n\n"
        f"{listed}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a saved model folder to the Hugging Face Hub.")
    parser.add_argument("--path", help="Explicit path to the model folder to upload.")
    parser.add_argument("--find-svm", action="store_true", help="Auto-locate an SVM router folder.")
    parser.add_argument("--model", help="Model family filter for auto-locate (berel/dictabert/hero/alephbertgimmel).")
    parser.add_argument("--condition", help="Condition-key substring filter for auto-locate.")
    parser.add_argument("--seed", help="Seed filter for auto-locate (e.g. 42).")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Upload every matching folder into per-condition subfolders of one repo.")
    parser.add_argument("--repo-id", required=True, help="Target HF repo. A bare name is prefixed with '%s/', e.g. 'berel-svm-fusion' -> '%s/berel-svm-fusion'." % (DEFAULT_HF_OWNER, DEFAULT_HF_OWNER))
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--commit-message", default="Upload thesis NER model", help="Commit message.")
    args = parser.parse_args()

    repo_id = args.repo_id if "/" in args.repo_id else f"{DEFAULT_HF_OWNER}/{args.repo_id}"

    folders = _find_folders(args)
    multi = len(folders) > 1
    is_svm = bool(args.find_svm) or all("svm_router" in f.name for f in folders)
    print(f"[upload] Repo   : {repo_id} ({'private' if args.private else 'public'})")
    for f in folders:
        print(f"[upload] Folder : {f}" + (f"  ->  {f.name}/" if multi else ""))

    from huggingface_hub import HfApi

    token = _resolve_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    for f in folders:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(f),
            # Keep each condition separate when uploading several at once.
            path_in_repo=(f.name if multi else "."),
            commit_message=args.commit_message,
        )

    # Upload a valid model card at the repo root (fixes the empty-metadata warning).
    api.upload_file(
        path_or_fileobj=_model_card(repo_id, folders, is_svm).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card metadata",
    )
    print(f"[done] Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
