#!/bin/bash
set -euo pipefail

rm -f examples.log

# Cartelle base
EXAMPLES_DIR="examples"
OUT_BASE="temp"

# Crea la cartella temp se non esiste
mkdir -p "$OUT_BASE"

# Scorri tutte le directory direttamente sotto examples/
# -print0 + read -d '' per gestire spazi e caratteri strani nei nomi
find "$EXAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' dir; do
  # nome directory (relativa rispetto a examples/)
  rel="${dir#"$EXAMPLES_DIR"/}"

  out_dir="$OUT_BASE/$rel/"

  echo "============================================================" | tee -a examples.log
  echo "[INFO] Directory trovata: $dir" | tee -a examples.log
  echo "[INFO] Output dir:       $out_dir" | tee -a examples.log

  # opzionale: pulisci l'output precedente di quella directory
  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  if [ -n "${GEMINI_API_KEY:-}" ]; then
    echo "[INFO] GEMINI_API_KEY=$GEMINI_API_KEY." | tee -a examples.log
    echo ./pdf2tree.sh --from-dir "$dir" --to-dir "$out_dir" --post-processing --verbose --debug | tee -a examples.log
    ./html2tree.sh --from-dir "$dir" --to-dir "$out_dir" --post-processing --verbose --debug >>examples.log 2>&1 && echo "[OK] on dir $dir" | tee -a examples.log || { rc=$?; echo "[ERROR] (rc=$rc) on dir $dir" | tee -a examples.log; continue; }
  else
    echo '[WARNING] GEMINI_API_KEY non definita. Aggiungerla con GEMINI_API_KEY=$(cat .gemini-api-key) ./html2tree.sh ...' | tee -a examples.log
    echo ./html2tree.sh --from-dir "$dir" --to-dir "$out_dir" --post-processing --disable-annotate-images --verbose | tee -a examples.log
    ./html2tree.sh --from-dir "$dir" --to-dir "$out_dir" --post-processing --disable-annotate-images --verbose --debug >>examples.log 2>&1 && echo "[OK] on dir $dir" | tee -a examples.log || { rc=$?; echo "[ERROR] (rc=$rc) on dir $dir" | tee -a examples.log; continue; }
  fi
done
