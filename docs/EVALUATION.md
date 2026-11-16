# Offline evaluation

Reproduce: `python manage.py evaluate_recommender` (about 2 minutes; writes `docs/eval_results.json`, the
raw source of every number below). Settings: catalogue of 43,998 tracks (after de-duplication), k = 10,
2,000 random query tracks (seed 0), K-Means sweep k = 2..12 (seed 42).

## Caveat

The dataset has no listening history, ratings or genre labels, so **none of these numbers measure user
satisfaction**. They are content-based proxies, each compared against a baseline. Treat them as sanity
checks and trade-off measurements, not as accuracy.

## Neighbour quality (cosine k-NN, k = 10)

Queries are tracks whose primary artist has at least one other track in the catalogue.

| Metric | k-NN | Random | Top-popularity |
|---|---|---|---|
| Artist hit-rate@10 (another track by same primary artist in list) | 0.565 | 0.165 | 0.011 |
| Language precision@10 (same language label as query) | 0.441 | 0.233 | 0.328 |
| Intra-list diversity (mean pairwise cosine distance) | 0.062 | 0.987 | 0.784 |
| Artist diversity (distinct primary artists / 10) | 0.677 | 0.922 | 0.600 |
| Catalogue coverage (share of tracks ever recommended) | 0.356 | 0.364 | 0.0002 |

Reading:
- k-NN recovers same-artist tracks about 3.4x as often as random (0.565 vs 0.165) and finds same-language
  tracks far more often than random, using audio features alone (language is never an input).
- The cost is diversity: lists are near-duplicates in feature space (ILD 0.062) and about a third of the
  slots repeat an artist. A re-ranker (e.g. MMR or an artist cap) is the obvious next step; it is not implemented.
- Language precision of 0.441 is modest: audio features only partially separate languages.
- Random coverage of 0.364 is only the result of 2,000 lists x 10 draws; coverage is not comparable across
  methods at this query count beyond showing that popularity recommends almost nothing.

## K-Means model selection (standardised 10 features, sampled metrics on 10,000 tracks)

| k | Silhouette (higher better) | Davies-Bouldin (lower better) | Inertia |
|---|---|---|---|
| 2 | 0.273 | 1.498 | 350,953 |
| 3 | 0.266 | 1.784 | 322,274 |
| 4 | 0.217 | 1.777 | 306,828 |
| 5 | 0.216 | 1.455 | 267,915 |
| 6 | 0.221 | 1.419 | 253,779 |
| 7 | 0.195 | 1.458 | 240,936 |
| 8 | 0.181 | 1.420 | 193,378 |
| 9 | 0.187 | 1.355 | 181,392 |
| 10 | 0.169 | 1.447 | 176,462 |
| 11 | 0.162 | 1.447 | 169,528 |
| 12 | 0.166 | 1.375 | 155,218 |

Findings:
- Silhouette is highest at k = 2 (0.273) and decreases overall. Values of 0.17 to 0.27 indicate weak
  cluster structure: audio features form a continuum, not clearly separated moods.
- k = 6 (the shipped default) is a local silhouette bump (0.221 vs 0.216 at k = 5) and has a lower
  Davies-Bouldin score than k = 2..5, but it was chosen as a compromise between granularity and these
  scores, not because it is statistically best. Davies-Bouldin is best at k = 9 (1.355).
- Cluster sizes at k = 6 are very uneven (917 to 18,866 tracks), and the heuristic mood labels (valence/energy
  of the centre vs the dataset mean) are not unique: three clusters are labelled "Melancholic / Low Energy".
  Mood labels are descriptive, not validated against human judgement.

## Limitations

- No ground truth for "good recommendation"; artist/language agreement is a proxy and penalises
  recommendations that are musically fine but by a different artist.
- Language mix in the raw CSV: English 37.5%, Unknown 20.9%, Tamil 20.3%, Korean 11.1%, Hindi 9.2%. "Unknown" is treated as a language label, which inflates random/popularity baselines slightly and makes the metric noisy for those queries.
- Silhouette/Davies-Bouldin use a 10,000-track subsample; inertia uses all tracks.
- Single seed, no confidence intervals.
