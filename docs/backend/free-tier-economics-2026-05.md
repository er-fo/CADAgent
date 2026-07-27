# Free-Tier Economics Baseline

## Purpose

This note captures the current cost model for a CADAgent free tier so the team can
validate that a Bedrock-backed launch is economically viable before replacing the
legacy iteration framing with a token budget.

## Data source

- Local checkout had no usable `runs/` data.
- Production legacy backend data was sampled from EC2 host `i-007470a13ac95c876`.
- Live run log root used for analysis: `/opt/backend-legacy/runs`.
- Sample window observed in the folder names: `2026-04-09` through `2026-04-30`.

## Observed usage shape from production runs

- Session directories: `183`
- Iterations: `857`
- Average iterations per session: `7.72`
- Median iterations per session: `4`
- Average model calls per iteration: `1.0`
- Average tokens per iteration: `3,761` input + `452` output
- Average total tokens per iteration: about `4,213`
- Recent activity spike: `833` iterations across the last `7` sampled days (`119/day`)

These logs were dominated by GPT and Claude traffic. The free-tier estimate below
reuses the observed token mix, then prices that token mix against Bedrock models.

## Bedrock pricing assumptions used

Conservative pricing should use the higher regional rates that match the AWS account
region posture rather than the cheapest global listing.

| Model | Input / 1M | Output / 1M | Pricing basis |
| --- | ---: | ---: | --- |
| MiniMax M2.5 | `$0.36` | `$1.44` | eu-north-1 style regional rate |
| Kimi K2.5 | `$0.72` | `$3.60` | eu-north-1 style regional rate |

Lower global or us-east rates also exist, but this document uses the higher rates as
the planning baseline.

## Estimated unit economics

Using the observed CADAgent token mix:

| Model | Cost per iteration |
| --- | ---: |
| MiniMax M2.5 | `$0.00200` |
| Kimi K2.5 | `$0.00433` |

Equivalent free-tier bundles at current usage shape:

| Free allowance | Approx token budget | MiniMax M2.5 | Kimi K2.5 |
| --- | ---: | ---: | ---: |
| 100 iteration-equivalents | `~421k` total tokens | `$0.20` | `$0.43` |
| 150 iteration-equivalents | `~632k` total tokens | `$0.30` | `$0.65` |
| 200 iteration-equivalents | `~843k` total tokens | `$0.40` | `$0.87` |

## Runway model

- Available AWS credits assumed: `$700`
- Required runway assumed: `1.5 months` (`45` days)
- Stress assumption: usage grows to `5x` the recent observed rate

Projected 45-day burn at `5x` the recent observed run volume:

| Model | 45-day burn |
| --- | ---: |
| MiniMax M2.5 | `$53.67` |
| Kimi K2.5 | `$116.04` |

This leaves large headroom against `$700`, so the bottleneck is not model inference
cost at the current CADAgent token shape.

## Recommendation

- Launch with a free cap of **150 iteration-equivalents per user**, or the token-based
  equivalent of roughly **600k total tokens per user**.
- If a more generous launch is desired, **200 iteration-equivalents** is still cheap.
- Treat **Kimi K2.5 regional pricing** as the conservative planning baseline.
- Revisit the cap after the first real free-tier cohort because user behavior may
  become more verbose once the product no longer exposes iteration counting directly.

## Validation notes

- Re-run the economics check from `/opt/backend-legacy/runs` after any material prompt,
  tool, or model-routing change.
- Confirm Bedrock pricing in AWS before launch in case regional pricing changes.
- Prefer using real Bedrock billing once production traffic has moved off GPT/Claude;
  this document is a token-shape projection, not a Bedrock invoice snapshot.
