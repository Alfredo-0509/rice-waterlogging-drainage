.PHONY: demo figures streamlit install

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

demo: $(VENV)/bin/python
	$(PY) run_demo.py

$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

figures: demo

streamlit: demo
	$(VENV)/bin/streamlit run app/dashboard.py
