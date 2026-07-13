# TMLR anonymous manuscript

This directory uses the official TMLR LaTeX style downloaded from:

- https://github.com/JmlrOrg/tmlr-style-file
- upstream commit: `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`
- retrieved: 2026-07-11

The files `tmlr.sty`, `tmlr.bst`, and `math_commands.tex` are copied without local style modifications. See `TMLR_STYLE_LICENSE` for the upstream license.

## Build

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Headline-result macros remain `PENDING` until completed raw metrics are archived and verified. This draft is not submission-ready while any pending cell remains.
