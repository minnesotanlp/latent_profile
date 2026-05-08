# Local Viewing

From the repo root:

```bash
cd docs
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/
```

## Structure

- `index.html` now follows the layout pattern of the Academic Project Page Template.
- Template-style assets live under:
  - `static/css`
  - `static/js`
  - `static/images`
  - `static/pdfs`

## Notes

- The page content and figures use only files referenced by `Latent_Profiles_in_LLM_Simulation__ICLR_2026_/main_arxiv.tex` or its dependents.
- Redundant drafts and duplicate figure exports in the paper bundle were not used.
- The site is mostly self-contained, but it still loads Google Fonts, Bulma, Font Awesome, and Academicons from CDNs.
- With no internet connection, the page will still load, but typography and icon styling may degrade.
- `.nojekyll` is present so GitHub Pages serves this directory as a static site instead of running the Jekyll build pipeline.
