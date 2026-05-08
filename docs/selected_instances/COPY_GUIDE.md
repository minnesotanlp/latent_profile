# Copy Guide

## Paper

Use the judge-reasonable examples as **comparisons**, not as isolated examples. The generated LaTeX follows the paper's current qualitative-box style: pipeline-stage title, pass/fail badge, compact A1/A2 metadata, agreement trajectory, and gray case separators.

- Main body: add one compact comparative box for Finding 4 after the robustness table discussion in `paper_latex/tex/4_experiments.tex`.
- Appendix: replace `paper_latex/tex/6_qualitative.tex` with `paper_appendix_d.tex`, or use `paper_snippets.tex` if you only want the generated subsections without the section heading.

Recommended paper examples:

- Main body: `paper_main_finding_04.tex`, because it cleanly compares low-contentious/high-agreement against high-contentious/low-agreement while holding `P=(1,1)` fixed.
- Appendix: `paper_appendix_d.tex` includes complete comparison boxes for all six findings.

All six finding groups are complete. Every displayed case has `judge_reasonable=true`, `display_ready=true`, and `near_duplicate_turns=0`.

## Website

Copy the whole bundle, but use `website_comparisons.json` as the primary input data. Use `website_examples.jsonl` only if you also want a flat index/search view.

Suggested webpage component:

- Left/right carousel through comparison groups.
- Optional finding filter or tabs.
- Side-by-side case cards within each comparison group.
- Within each case card, side-by-side Agent 1 / Agent 2 transcript columns.
- Metadata header showing `P`, `O`, `A`, topic, model, finding, and judge reasonableness.
- Collapsible provenance details.

## Transfer

Copy `selected_instances_bundle.tar.gz` to the webpage machine, then unpack it:

```bash
tar -xzf selected_instances_bundle.tar.gz
```
