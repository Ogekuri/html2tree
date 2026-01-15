#!/bin/bash
set -euo pipefail

rm -f html_sample.log

rm -rf temp/html_sample_test/

# Verifica se la variabile GEMINI_API_KEY è definita e non vuota
if [ -n "${GEMINI_API_KEY:-}" ]; then
    echo "[INFO] GEMINI_API_KEY=$GEMINI_API_KEY." | tee -a html_sample.log
    echo ./http2tree.sh --from-dir html_sample/ --to-dir temp/html_sample_test/ --verbose --debug | tee -a html_sample.log
    ./http2tree.sh --from-dir html_sample/ --to-dir temp/html_sample_test/ --verbose --debug >>html_sample.log 2>&1 && echo "[OK] on dir html_sample" | tee -a html_sample.log || { rc=$?; echo "[ERROR] (rc=$rc) on dir html_sample" | tee -a html_sample.log; continue; }
else
    echo '[WARNING] GEMINI_API_KEY non definita. Aggiungerla con GEMINI_API_KEY=$(cat .gemini-api-key) ./pdf2tree.sh ...' | tee -a html_sample.log
    echo ./http2tree.sh --from-dir html_sample/ --to-dir temp/html_sample_test/ --verbose --debug | tee -a html_sample.log
    ./http2tree.sh --from-dir html_sample/ --to-dir temp/html_sample_test/ --verbose --debug >>html_sample.log 2>&1 && echo "[OK] on dir html_sample" | tee -a html_sample.log || { rc=$?; echo "[ERROR] (rc=$rc) on dir html_sample" | tee -a html_sample.log; continue; }
fi



