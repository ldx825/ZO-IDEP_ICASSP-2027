MAIN=main

.PHONY: all build clean watch check experiments

all: build

build:
	latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error $(MAIN).tex

watch:
	latexmk -pvc -pdf -synctex=1 -interaction=nonstopmode -file-line-error $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

check: build
	bash scripts/check_submission.sh $(MAIN).pdf

experiments:
	python3 zo_idep/experiments/exp01_sparse_regression.py --trials 30 --iterations 300

#	The legacy scripts/ pipeline is deprecated; see scripts/DEPRECATED.md.
#	Canonical experiments: zo_idep/experiments/exp01--exp07 (see EXPERIMENT_CHECKLIST.md).
