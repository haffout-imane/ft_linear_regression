PYTHON ?= python

.PHONY: help install all train predict plot precision clean

help:
	@echo "Targets: install, train, predict, plot, precision, all, clean"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train.py

predict:
	$(PYTHON) predict.py

plot:
	$(PYTHON) plot.py

precision:
	$(PYTHON) precision.py

all: train
	$(PYTHON) plot.py
	$(PYTHON) precision.py

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; shutil.rmtree('__pycache__', ignore_errors=True); shutil.rmtree('.pytest_cache', ignore_errors=True); Path('thetas.json').unlink(missing_ok=True); Path('plot.png').unlink(missing_ok=True)"
