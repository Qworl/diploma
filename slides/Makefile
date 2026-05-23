.PHONY: pdf clean watch rebuild

SHELL := /bin/bash
PATH := /Library/TeX/texbin:$(PATH)
export PATH

pdf:
	PATH=/Library/TeX/texbin:$$PATH latexmk -xelatex main.tex

rebuild: clean pdf

clean:
	PATH=/Library/TeX/texbin:$$PATH latexmk -C

watch:
	PATH=/Library/TeX/texbin:$$PATH latexmk -pvc -xelatex main.tex
