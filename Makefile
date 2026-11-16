.PHONY: install demo all-models gui check streamlit figures clean

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install: $(VENV)/bin/python

$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

demo: $(VENV)/bin/python
	$(PY) run_demo.py

# Compare all three hydrology models (bucket / two_layer / plough_pan).
all-models: $(VENV)/bin/python
	$(PY) run_demo.py --all-models

# Tkinter control room: one button per step, one tab per CSV.
gui: $(VENV)/bin/python
	$(PY) app/gui.py

# Headless import / artefact check, no window needed.
check: $(VENV)/bin/python
	$(PY) app/gui.py --check

figures: demo

streamlit: demo
	$(VENV)/bin/streamlit run app/dashboard.py

clean:
	rm -rf __pycache__ app/__pycache__ scripts/__pycache__ src/__pycache__
