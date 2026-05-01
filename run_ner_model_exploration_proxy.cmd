@echo off

REM --------- Easy-to-edit runtime settings ---------
set "THESIS_PREP_PROFILE=full"
REM Options: ultra_fast | smoke | full
set "THESIS_PREP_REBUILD_DATASET_CACHE=1"

REM --------- Network / proxy settings ---------
set "THESIS_DISABLE_SSL_VERIFY=1"
set "http_proxy=http://proxy-dmz.intel.com:912"
set "https_proxy=http://proxy-dmz.intel.com:912"
set "no_proxy=localhost,intel.com,127.0.0.1"
set "THESIS_HTTP_PROXY=http://proxy-dmz.intel.com:912"
set "THESIS_HTTPS_PROXY=http://proxy-dmz.intel.com:912"
set "THESIS_NO_PROXY=localhost,intel.com,127.0.0.1"

REM --------- Augmentation model settings (for exp08 / exp07+aug) ---------
REM Leave unset to use code defaults:
REM 1) prefer local MLM cache for matching model family (DictaBERT / BEREL)
REM 2) if unavailable, fall back to hub download
set "THESIS_AUGMENTATION_LOCAL_ONLY="
set "THESIS_AUGMENTATION_MODEL_NAME="

".\.venv\Scripts\python.exe" "prep\ner_model_exploration.py"