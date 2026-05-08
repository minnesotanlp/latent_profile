# Webpage Integration Prompt

You are integrating qualitative examples from a behavioral-coherence paper into an existing project webpage.

## Bundle Contents

This bundle contains selected conversation examples from the latent-profile social simulation experiments.

- `website_comparisons.json`: primary carousel data. Each object is one finding-level comparison group with a claim, relationship summary, and two selected cases.
- `website_examples.jsonl`: flat per-example data for search, filtering, or debugging.
- `manifest.json`: complete provenance and grouped comparison metadata.
- `paper_snippets.tex`: all LaTeX snippets for paper use.
- `paper_appendix_d.tex`: self-contained replacement for Appendix D's qualitative-example section.
- `paper_main_finding_04.tex`: compact main-body comparative example for Finding 4.
- `finding_01.tex` through `finding_06.tex`: per-finding LaTeX snippets.
- `review_notes.md`: human-readable selection notes.

All exported examples in `website_examples.jsonl` have `judge_reasonable=true`, `paper_ready=true`, `display_ready=true`, and `near_duplicate_turns=0`. Every finding has a complete two-case comparison group.

## Webpage Goal

Build a polished, scrollable side-by-side comparison viewer. Users should move left/right through comparison groups using arrow buttons. Each view should show:

- Finding title, comparison title, claim, and relationship summary.
- Two side-by-side case cards when the comparison is complete.
- For each case: topic, contentiousness level, model, agent summaries, preference scores `P=(P1,P2)`, openness scores `O=(O1,O2)`, and agreement trajectory `A=(...)`.
- Within each case: two-column conversation transcript with Agent 1 turns on the left and Agent 2 turns on the right, aligned by round.
- A small provenance/details area with source path and bin/sample ids, preferably collapsible.

## Recommended UI Style

Use a restrained research-project style rather than a marketing page:

- Keep the viewer dense but readable.
- Use a tab or segmented control for findings if the page already has one; otherwise use left/right buttons for the full comparison sequence.
- Use color sparingly:
  - green or blue accent for high-agreement examples;
  - red or amber accent for low-agreement or failure examples;
  - neutral grays for metadata.
- Make the agreement trajectory visually scannable, for example small chips labeled `A1`, `A2`, `A3`, `A4` or a compact sparkline-like row.
- Do not hide the judge trajectory; it is central to why the examples were selected.
- Keep transcript text selectable and readable on mobile.

## Data Handling

Parse `website_comparisons.json` as the main data source. Each group has:

- `finding_id`
- `finding_title`
- `comparison_title`
- `claim`
- `relationship`
- `paper_priority`
- `website_priority`
- `is_complete`
- `cases`
- `unsupported_slots`

Each `cases[]` item uses the same fields as the flat `website_examples.jsonl` rows. The package has no unsupported slots; keep `unsupported_slots` support only as a defensive fallback for alternate data bundles.

Expected useful fields per row:

- `finding_id`
- `finding_title`
- `slot`
- `slot_label`
- `model_name`
- `topic_title`
- `contentiousness`
- `agent_1_summary`
- `agent_2_summary`
- `agent_1_topic_response`
- `agent_2_topic_response`
- `agent_1_persuadability`
- `agent_2_persuadability`
- `pref_gap`
- `combined_openness`
- `judge_scores`
- `judge_reasonableness_score`
- `judge_reasonableness_reason`
- `utterances`
- `source_path`
- `pair_bin_1`, `pair_bin_2`, `sample_idx`

## Important Interpretation Notes

These examples were selected to make judge scores readable from the transcript. They are not intended to be the most statistically extreme cases. For each example, the visible public turns should make the reported agreement trajectory plausible.

For transcript display inside each case, pair utterances as:

- Round 1: `utterances[0]` vs `utterances[1]`
- Round 2: `utterances[2]` vs `utterances[3]`

If more turns are added later, continue the same even/odd pairing convention.
