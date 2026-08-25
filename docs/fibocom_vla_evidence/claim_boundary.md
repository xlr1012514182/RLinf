# Claim-to-evidence boundary

This file prevents code capability, upstream results, historical experiments,
and new local measurements from being merged into one claim.

| Résumé statement | Source audit result | Implementation status | Result status |
|---|---|---|---|
| frozen π0.5 + 1.82M residual actor + twin Q | Local historical code freezes OpenPI, but its 1.82M figure is actor plus twin-Q, not actor alone | This branch constructs and asserts a 1.82M actor, with twin-Q separate, to follow the requested résumé wording | No new policy-quality result yet |
| binary sparse reward, no shaping, second-failure tail, GAE, one PPO epoch | Historical `stage_binary_guard` contains geometric shaping; historical PPO mixes non-policy actions and uses a questionable Q-as-V/log-prob path | This branch uses binary task outcomes, an explicit second-failure terminal transform, masked GAE, true bounded-policy log-prob, and a single PPO epoch | Unit-tested only |
| 112 self-rollouts, 75.0%→96.9%, reduced drops on a real robot | Audited local artifacts contain 80+32 episodes, 96/128→124/128, and 30→2 hit/drop events, but the runs use different seeds, reference guards/rescue, and IsaacLab/Franka simulation | Archive/collector interfaces are prepared; no result is hard-coded | Not verified as paired or physical-robot evidence |
| 48.2→42.8 ms CUDA Graph, bit-exact | Historical evidence closes 48.1572→42.7936 ms on a StarVLA/Qwen visual path | Shape-gated capture and strict exactness checks are implemented | Not yet reproduced on π0.5 |
| TensorRT 1.5×, BF16 fusion rounding | Historical audit does not close a π0.5 1.5× deployment result | Paired runner and layer-diff diagnostics are provided; BF16 fusion is an evidence hypothesis, never an automatic conclusion | Not verified |
| continuous speculative 58.0→19.1 ms, 3.04×, 94.1→93.8 | Those values are attributable to upstream FLASH material | B×K verification/prefix acceptance is adapted; résumé-specific same-round takeover is labelled as a deliberate override from upstream next-round fallback | Upstream-only until reproduced here |
| RTC 5.6 s→4 ms and 76→97 ms | 76→97 ms is published model-compute evidence; 5.6 s→4 ms cannot be called model latency without a clock definition | Native denoising guidance, hard prefix, async scheduling, and separate compute/wait/loop metrics are implemented | No local GPU/robot result yet |

The current maximum defensible outcome is component-level/Mock verification.
Physical success rate, π0.5 latency, TensorRT speedup, and end-to-end RTC quality
remain blocked on the model checkpoint, task data, GPU protocol, and robot
integration described in the run manifests.
