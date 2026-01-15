#!/bin/bash
set -euo pipefail

rm -f html_sample.log

# Verifica se la variabile GEMINI_API_KEY è definita e non vuota
if [ -n "${GEMINI_API_KEY:-}" ]; then
    echo "[INFO] GEMINI_API_KEY=$GEMINI_API_KEY." | tee -a html_sample.log
    echo ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --post-processing-only --disable-toc --verbose --debug | tee -a html_sample.log
    ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --post-processing-only --disable-toc --verbose --debug >>html_sample.log 2>&1 && echo "[OK] on dir html_sample" | tee -a html_sample.log || { rc=$?; echo "[ERROR] (rc=$rc) on dir html_sample" | tee -a html_sample.log; continue; }
else
    echo '[WARNING] GEMINI_API_KEY non definita. Aggiungerla con GEMINI_API_KEY=$(cat .gemini-api-key) ./pdf2tree.sh ...' | tee -a html_sample.log
    echo ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --post-processing-only --disable-annotate-images --disable-toc --verbose --debug | tee -a html_sample.log
    ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --post-processing-only --disable-annotate-images --disable-toc --verbose --debug >>html_sample.log 2>&1 && echo "[OK] on dir html_sample" | tee -a html_sample.log || { rc=$?; echo "[ERROR] (rc=$rc) on dir html_sample" | tee -a html_sample.log; continue; }
fi
