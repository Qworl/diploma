.PHONY: pdf clean watch rebuild

# Гарантируем доступность xelatex/biber/latexmk даже если PATH «старой» оболочки
# не подхватил basictex (basictex кладёт симлинки в /Library/TeX/texbin/, которые
# подключает /etc/paths.d/TeX — но только в новых терминалах).
SHELL := /bin/bash
PATH := /Library/TeX/texbin:$(PATH)
export PATH

# Сборка PDF (xelatex + biber через latexmk; настройки в .latexmkrc).
pdf:
	PATH=/Library/TeX/texbin:$$PATH latexmk -xelatex main.tex

# Чистая пересборка
rebuild: clean pdf

clean:
	PATH=/Library/TeX/texbin:$$PATH latexmk -C
	rm -f *.bbl *.bcf *.run.xml *.glsdefs *.glo *.gls *.glg *.ist *.acn *.acr *.alg

# Просмотр с авто-пересборкой
watch:
	PATH=/Library/TeX/texbin:$$PATH latexmk -pvc -xelatex main.tex
