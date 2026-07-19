.PHONY: all results manuscript test clean

all: results manuscript

results:
	python scripts/run_all.py

manuscript:
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

test:
	python -m pytest -q

clean:
	cd manuscript && latexmk -C
	rm -rf results figures/*.png figures/*.pdf data/demo/*
