#!/usr/bin/env bash
# Serial baseline: SuiteSparse CSR, one dense matvec, single-thread roofline.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "${root}"

if [[ ! -x "${root}/build/spmv" ]]; then
  echo "build the Release binary first: cmake --build build -j" >&2
  exit 1
fi

bash "${root}/scripts/download_matrices.sh"
mkdir -p "${root}/results" "${root}/plots"
rm -f "${root}/results/m1.csv" "${root}/results/machine.csv" "${root}/results/table1.csv"

echo "measure roofs"
"${root}/build/spmv" --machine --csv-out "${root}/results/machine.csv"

while read -r group name rest; do
  if [[ -z "${group}" || "${group}" == \#* ]]; then
    continue
  fi
  echo "spmv ${name}"
  "${root}/build/spmv" "${root}/data/${name}.mtx" --csv-out "${root}/results/m1.csv"
done < "${root}/scripts/matrices.txt"

echo "dense 4096"
"${root}/build/spmv" --dense 4096 --csv-out "${root}/results/m1.csv"

python3 "${root}/scripts/plot_gflops.py" "${root}/results/m1.csv" "${root}/plots/gflops.png"
python3 "${root}/scripts/plot_roofline.py" "${root}/results/m1.csv" "${root}/results/machine.csv" "${root}/plots/roofline.png"
python3 "${root}/scripts/table_m1.py" "${root}/results/m1.csv" "${root}/results/table1.csv"
