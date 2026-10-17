#!/bin/sh
# Архив проекта для сервера: код, обученные классификаторы, документы. Без датасета, кэшей и venv.
#   sh deploy/make_bundle.sh   ->   dist/cashguard_bundle.tar.gz
set -eu
cd "$(dirname "$0")/.."
mkdir -p dist
COPYFILE_DISABLE=1 tar --no-xattrs -czf dist/cashguard_bundle.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' \
  cashguard service scripts tools deploy labels docs \
  models/safe_clf.joblib models/money_clf.joblib \
  requirements-service.txt
ls -lh dist/cashguard_bundle.tar.gz
