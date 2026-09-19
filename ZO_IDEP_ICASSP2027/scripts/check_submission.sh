#!/usr/bin/env bash
set -euo pipefail

pdf_path="${1:-main.pdf}"

if [[ ! -f "$pdf_path" ]]; then
  echo "ERROR: $pdf_path does not exist."
  exit 1
fi

echo "== ICASSP 2027 PDF preflight =="

size_bytes=$(wc -c < "$pdf_path" | tr -d ' ')
max_bytes=$((5 * 1024 * 1024))
echo "File size: $size_bytes bytes"
if (( size_bytes > max_bytes )); then
  echo "FAIL: PDF is larger than 5 MB."
  exit 1
fi

pages=$(pdfinfo "$pdf_path" | awk '/^Pages:/ {print $2}')
echo "Pages: ${pages:-unknown}"
if [[ -z "$pages" ]] || (( pages > 5 )); then
  echo "FAIL: Page count is unavailable or exceeds five."
  exit 1
fi

pdfinfo "$pdf_path" | grep -E '^Page size:' || true

font_report=$(mktemp)
trap 'rm -f "$font_report"' EXIT
pdffonts "$pdf_path" | tee "$font_report"

if tail -n +3 "$font_report" | grep -q 'Type 3'; then
  echo "FAIL: Type 3 font detected."
  exit 1
fi
echo "PASS: No Type 3 fonts."

if tail -n +3 "$font_report" | grep -Eq '[[:space:]]no[[:space:]]+(yes|no)[[:space:]]+(yes|no)[[:space:]]+[0-9]'; then
  echo "FAIL: One or more fonts are not embedded."
  exit 1
fi
echo "PASS: Fonts are embedded."

if (( pages == 5 )); then
  page_five=$(pdftotext -f 5 -l 5 "$pdf_path" -)
  if ! grep -qi 'references' <<<"$page_five"; then
    echo "WARN: Page 5 does not visibly contain a References heading; inspect manually."
  fi
fi

echo "Basic automated preflight passed."
echo "Manual checks still required: author metadata and page-5 content boundaries."
