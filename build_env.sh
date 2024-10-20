virtualenv --python=$(which python) .venv
source .venv/bin/activate
pip install --upgrade pip
pip install poetry
git submodule init && git submodule update
poetry config virtualenvs.options.always-copy true
poetry config virtualenvs.in-project true
poetry install -v
