# Selected Conversation Instances

- Source root: `/space3/sciphi/latent_profile_outputs/core_fresh`
- Focal model preference: `8`
- Clean candidates scanned: `608766`

## Finding 1: Preference Gap Decreases Agreement
- Comparison claim: Pairs with the same elicited preference should agree more than pairs at the maximum preference gap.
- Relationship: Compare the aligned high-agreement case against the opposed low-agreement case.
- Aligned preference, high agreement: google/gemma-3-12b-it on Taxes from `/space3/sciphi/latent_profile_outputs/core_fresh/8/0/1` bins (10, 8, 15). P=(5,5), O=(6,3), A=(5,5,5,5). Judge-reasonable=True (score=11): final turns share a stance; judge trajectory is mostly high; no very-low judge score.
- Opposed preference, low agreement: google/gemma-3-12b-it on Spring vs. Fall from `/space3/sciphi/latent_profile_outputs/core_fresh/8/6/8` bins (0, 9, 4). P=(1,5), O=(2,5), A=(1,1,1,1). Judge-reasonable=True (score=13): final turns maintain incompatible stances; judge trajectory is mostly low; one side is softer while the other remains firm; no high judge score.

## Finding 2: Bias Instruction Asymmetry
- Comparison claim: High-bias instructions produce strong agreement when preferences align and strong disagreement when preferences oppose.
- Relationship: Compare a high-bias aligned high-agreement pair against a high-bias opposed low-agreement pair.
- High-bias aligned pair: google/gemma-3-12b-it on Taxes from `/space3/sciphi/latent_profile_outputs/core_fresh/8/0/8` bins (3, 3, 1). P=(5,5), O=(2,2), A=(5,5,5,5). Judge-reasonable=True (score=11): final turns share a stance; judge trajectory is mostly high; no very-low judge score.
- High-bias opposed pair with low agreement: google/gemma-3-12b-it on Coca-Cola vs. Pepsi from `/space3/sciphi/latent_profile_outputs/core_fresh/8/8/8` bins (0, 5, 8). P=(1,5), O=(3,3), A=(1,1,1,1). Judge-reasonable=True (score=13): final turns maintain incompatible stances; judge trajectory is mostly low; one side is softer while the other remains firm; no high judge score.

## Finding 3: Shared Sentiment, Divergent Destinations
- Comparison claim: The (1,1) and (5,5) shared-preference cases can yield different agreement trajectories despite both having zero preference gap.
- Relationship: Compare shared negative low agreement against shared positive high agreement.
- Shared negative pair with low agreement: google/gemma-3-12b-it on Spring vs. Fall from `/space3/sciphi/latent_profile_outputs/core_fresh/8/6/8` bins (3, 4, 10). P=(1,1), O=(5,6), A=(2,2,2,2). Judge-reasonable=True (score=13): final turns maintain incompatible stances; judge trajectory is mostly low; one side is softer while the other remains firm; no high judge score.
- Shared positive pair with high agreement: google/gemma-3-12b-it on Taxes from `/space3/sciphi/latent_profile_outputs/core_fresh/8/0/2` bins (10, 5, 6). P=(5,5), O=(6,4), A=(5,5,5,5). Judge-reasonable=True (score=11): final turns share a stance; judge trajectory is mostly high; no very-low judge score.

## Finding 4: Contentiousness at Shared Preference
- Comparison claim: Holding P=(1,1) fixed, low-contentious topics can show high agreement while high-contentious topics can show low agreement.
- Relationship: Compare low-contentious high agreement against high-contentious low agreement.
- Low-contentiousness shared-negative pair: google/gemma-3-12b-it on Beaches vs. Mountains from `/space3/sciphi/latent_profile_outputs/core_fresh/8/7/4` bins (0, 3, 11). P=(1,1), O=(3,6), A=(4,5,5,5). Judge-reasonable=True (score=11): final turns share a stance; judge trajectory is mostly high; no very-low judge score.
- High-contentiousness shared-negative pair: meta-llama/Llama-3.2-1B-Instruct on Immigration from `/space3/sciphi/latent_profile_outputs/core_fresh/3/1/5` bins (1, 1, 18). P=(1,1), O=(3,1), A=(2,1,1,1). Judge-reasonable=True (score=13): final turns maintain incompatible stances; judge trajectory is mostly low; one side is softer while the other remains firm; no high judge score.

## Finding 5: Openness Increases Agreement
- Comparison claim: Low-openness pairs can remain stuck in disagreement, while high-openness pairs more readily reach visible agreement.
- Relationship: Compare a low-openness low-agreement case against a high-openness high-agreement case.
- Low-openness pair with low agreement: meta-llama/Llama-3.2-3B-Instruct on Coca-Cola vs. Pepsi from `/space3/sciphi/latent_profile_outputs/core_fresh/4/8/4` bins (5, 17, 16). P=(2,5), O=(1,0), A=(1,1,1,1). Judge-reasonable=True (score=13): final turns maintain incompatible stances; judge trajectory is mostly low; one side is softer while the other remains firm; no high judge score.
- High-openness pair with high agreement: google/gemma-3-12b-it on Beaches vs. Mountains from `/space3/sciphi/latent_profile_outputs/core_fresh/8/7/3` bins (12, 19, 2). P=(5,5), O=(8,8), A=(5,5,5,5). Judge-reasonable=True (score=11): final turns share a stance; judge trajectory is mostly high; no very-low judge score.

## Finding 6: Low Openness and High Gap Produces Lowest Agreement
- Comparison claim: Among maximum preference-gap pairs, low openness can produce low agreement while high openness can still allow visible convergence.
- Relationship: Compare a low-openness max-gap low-agreement pair against a high-openness max-gap high-agreement pair.
- Low-openness max-gap pair with low agreement: google/gemma-3-12b-it on Spring vs. Fall from `/space3/sciphi/latent_profile_outputs/core_fresh/8/6/8` bins (0, 6, 5). P=(1,5), O=(2,2), A=(2,2,1,1). Judge-reasonable=True (score=13): final turns maintain incompatible stances; judge trajectory is mostly low; one side is softer while the other remains firm; no high judge score.
- High-openness max-gap pair with high agreement: meta-llama/Llama-3.1-8B-Instruct on Beaches vs. Mountains from `/space3/sciphi/latent_profile_outputs/core_fresh/5/7/3` bins (3, 7, 18). P=(1,5), O=(9,9), A=(4,4,4,5). Judge-reasonable=True (score=11): final turns share a stance; judge trajectory is mostly high; no very-low judge score.
