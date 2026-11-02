#!/usr/bin/env bash
# Download the Milestone 1 SuiteSparse set into data/<name>.mtx.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/data"
list="$root/scripts/matrices.txt"
tmp=

cleanup() {
  if [[ -n "${tmp}" && -d "${tmp}" ]]; then
    rm -rf "${tmp}"
  fi
}
trap cleanup EXIT

while read -r group name rest; do
  if [[ -z "${group}" || "${group}" == \#* ]]; then
    continue
  fi
  if [[ -n "${rest}" ]]; then
    echo "bad line in ${list}: ${group} ${name} ${rest}" >&2
    exit 1
  fi
  dest="$root/data/${name}.mtx"
  if [[ -f "${dest}" ]]; then
    echo "ok ${dest}"
    continue
  fi
  tmp=$(mktemp -d)
  urls=(
    "https://sparse.tamu.edu/MM/${group}/${name}.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/${group}/${name}.tar.gz"
    "http://www.cise.ufl.edu/research/sparse/MM/${group}/${name}.tar.gz"
  )
  fetched=0
  for url in "${urls[@]}"; do
    echo "fetch ${url}"
    if curl -L --fail --retry 2 --retry-delay 2 -o "${tmp}/pkg.tar.gz" "${url}"; then
      fetched=1
      break
    fi
    echo "failed ${url}" >&2
  done
  if [[ "${fetched}" -ne 1 ]]; then
    echo "could not download ${group}/${name}" >&2
    exit 1
  fi
  tar -xzf "${tmp}/pkg.tar.gz" -C "${tmp}"
  mapfile -t found < <(find "${tmp}" -name '*.mtx' -type f)
  chosen=""
  if [[ ${#found[@]} -eq 1 ]]; then
    chosen="${found[0]}"
  else
    for path in "${found[@]}"; do
      if [[ "$(basename "${path}")" == "${name}.mtx" ]]; then
        chosen="${path}"
        break
      fi
    done
  fi
  if [[ -z "${chosen}" ]]; then
    echo "could not choose a .mtx for ${name}; found ${#found[@]}" >&2
    exit 1
  fi
  mv "${chosen}" "${dest}"
  rm -rf "${tmp}"
  tmp=
  echo "wrote ${dest}"
done < "${list}"
