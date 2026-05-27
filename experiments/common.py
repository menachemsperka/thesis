from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
DEBUG = False
DEFAULT_MODEL_ID = "dicta-il/dictabert"


def configure_network_environment() -> dict[str, str]:
    is_colab = os.environ.get("THESIS_RUN_ENV") == "colab"
    
    if is_colab:
        http_proxy = ""
        https_proxy = ""
    else:
        http_proxy = (
            os.environ.get("THESIS_HTTP_PROXY")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
            or "http://proxy-dmz.intel.com:912"
        )
        https_proxy = (
            os.environ.get("THESIS_HTTPS_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or "http://proxy-dmz.intel.com:912"
        )

    no_proxy = (
        os.environ.get("THESIS_NO_PROXY")
        or os.environ.get("NO_PROXY")
        or os.environ.get("no_proxy")
        or "localhost,intel.com,127.0.0.1"
    )

    if is_colab:
        # COLAB_FIXED_PROXY_DECL: http_proxy = "http://proxy-dmz.intel.com:912"
        # COLAB_FIXED_PROXY_ASSIGN: os.environ["HTTP_PROXY"] = http_proxy
        pass
    else:
        os.environ["HTTP_PROXY"] = http_proxy
        os.environ["http_proxy"] = http_proxy
        os.environ["HTTPS_PROXY"] = https_proxy
        os.environ["https_proxy"] = https_proxy
        
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy

    ca_bundle = (
        os.environ.get("THESIS_CA_BUNDLE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or ""
    ).strip()
    disable_ssl_verify = (os.environ.get("THESIS_DISABLE_SSL_VERIFY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["CURL_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle

    if disable_ssl_verify:
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["SSL_CERT_FILE"] = ""

    return {
        "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
        "NO_PROXY": os.environ.get("NO_PROXY", ""),
        "REQUESTS_CA_BUNDLE": os.environ.get("REQUESTS_CA_BUNDLE", ""),
        "THESIS_DISABLE_SSL_VERIFY": os.environ.get("THESIS_DISABLE_SSL_VERIFY", "0"),
    }


configure_network_environment()


def is_debug_enabled() -> bool:
    env_value = os.environ.get("THESIS_DEBUG", "0").strip().lower()
    return DEBUG or env_value in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def suppress_output_if_needed(debug: bool | None = None):
    use_debug = is_debug_enabled() if debug is None else debug
    if use_debug:
        yield
        return

    buffer_out = io.StringIO()
    buffer_err = io.StringIO()
    with contextlib.redirect_stdout(buffer_out), contextlib.redirect_stderr(buffer_err):
        yield


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_experiment_output_dir(experiment_id: str) -> Path:
    ensure_outputs_dir()
    exp_dir = OUTPUTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def resolve_dataset(filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def resolve_model_source() -> tuple[str, bool]:
    env_model = os.environ.get("THESIS_MODEL_NAME")
    if env_model:
        return env_model, Path(env_model).exists()

    internal_model = PROJECT_ROOT / "models" / "dictabert"
    if (internal_model / "config.json").exists() and (internal_model / "tokenizer.json").exists():
        return str(internal_model), True

    return DEFAULT_MODEL_ID, False


def configure_model_environment() -> tuple[str, bool]:
    model_name, is_local = resolve_model_source()
    os.environ["THESIS_MODEL_NAME"] = model_name
    os.environ["THESIS_MODEL_LOCAL_ONLY"] = "1" if is_local else "0"
    if is_local:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    return model_name, is_local


def write_result_json(experiment_id: str, base_name: str, payload: dict[str, Any]) -> Path:
    exp_dir = get_experiment_output_dir(experiment_id)
    out_path = exp_dir / f"{base_name}_{now_timestamp()}.json"
    payload["result_file"] = str(out_path)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_path = exp_dir / "latest.json"
    latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def write_result_excel(
    experiment_id: str,
    base_name: str,
    metrics_df,
    detailed_df,
    extra_sheets: dict[str, Any] | None = None,
) -> Path:
    import pandas as pd

    exp_dir = get_experiment_output_dir(experiment_id)
    out_path = exp_dir / f"{base_name}_{now_timestamp()}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if metrics_df is None:
            pd.DataFrame().to_excel(writer, sheet_name="metrics", index=False)
        else:
            metrics_df.to_excel(writer, sheet_name="metrics", index=False)

        if detailed_df is not None and not detailed_df.empty:
            detailed_df.to_excel(writer, sheet_name="detailed_results", index=False)

        if extra_sheets:
            for sheet_name, sheet_df in extra_sheets.items():
                if sheet_df is None:
                    continue
                normalized_name = str(sheet_name)[:31]
                if hasattr(sheet_df, "empty") and sheet_df.empty:
                    continue
                sheet_df.to_excel(writer, sheet_name=normalized_name, index=False)

    latest_path = exp_dir / f"{base_name}_latest.xlsx"
    if latest_path.exists():
        latest_path.unlink()
    shutil.copy2(out_path, latest_path)
    return out_path


def write_split_runs_excel(
    experiment_id: str,
    base_name: str,
    runs_df,
    summary_df=None,
) -> Path:
    import pandas as pd

    exp_dir = get_experiment_output_dir(experiment_id)
    out_path = exp_dir / f"{base_name}_{now_timestamp()}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if runs_df is None:
            pd.DataFrame().to_excel(writer, sheet_name="per_split", index=False)
        else:
            runs_df.to_excel(writer, sheet_name="per_split", index=False)

        if summary_df is not None:
            summary_df.to_excel(writer, sheet_name="summary", index=False)

    latest_path = exp_dir / f"{base_name}_latest.xlsx"
    if latest_path.exists():
        latest_path.unlink()
    shutil.copy2(out_path, latest_path)
    return out_path
