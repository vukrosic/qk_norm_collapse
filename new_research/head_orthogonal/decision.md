# Chief Scientist Decision: HeadOrtho

---

## 3 Arguments for REVISE

**R1: The mode-2 fix is genuinely elegant.**
The critic's strongest technical objection — that mode-1 produces a 128×32768 pathological matrix — was completely neutralized by the defender's mode-2 solution. Mode-2 gives `[2048, 2048]` for Q (perfect square) and `[2048, 1024]` for K (2:1 ratio). These are well within the operating range of Polar Express. This isn't patchwork; it's a strict improvement that the original Teon authors would likely endorse. The defender correctly identified that mode-2 is mechanistically more natural for attention heads (shared input space = shared left singular vectors).

**R2: The idea has a safety floor — it cannot be worse than Muon.**
Teon's Theorem 1 proves L_teon ≤ L_muon unconditionally. The √K speedup requires alignment, but the "never worse" guarantee does not. The defender correctly identified this asymmetry. If cross-head alignment is low, tensor orthogonalization degrades gracefully to something equivalent to per-head orthogonalization plus some rotational mixing. This means the experiment has bounded downside: the worst case is matching Muon, not catastrophic failure.

**R3: The revised plan is lean and executable.**
V1.1 has: one diagnostic (30 min), one smoke test (30 min), one training run (4–8 hrs), one comparison against existing baseline. Total active work: ~10 hours. Total wall time including buffer: well within 24 hours. The single conclusion is crisp: "Does val loss improve?" This is a real 24-hour research project now, not a scatter-shot exploration.

---

## 3 Arguments for DROP

**D1: The core analogy is dubious — layers ≠ heads.**
Teon orthogonalizes across layers of the *same type* (Q₆ with Q₇, K₆ with K₇). These are architecturally identical matrices processing the same data flow, just at different depths. Cross-layer gradient alignment is natural because adjacent layers see similar representations. Heads within the same layer are *architecturally identical but functionally different by design* — they are supposed to attend to different things. Orthogonalizing across heads may be fighting the model's natural specialization gradient. The Teon analogy is structurally misleading: it maps "different layers of the same type" to "different heads of the same projection," but these are very different kinds of redundancy.

**D2: No prior evidence that head gradient alignment exists.**
The defender's Step 0 diagnostic is the right idea, but the fact that it hasn't been done yet is concerning. The entire project rests on an unverified assumption. If the diagnostic shows cosine similarity < 0.3, the defender admits the Teon framework "fails" and proposes falling back to per-head orthogonalization — which is just Muon on smaller matrices. At that point, the research contribution evaporates: you're just reshaping the problem, not solving it.

**D3: The "val loss improvement" bar may be unreachable at 20M tokens.**
Teon shows improvements of 0.5–3 PPL points on models trained for 2–13B tokens. At 20M tokens, the model is in the high-loss, high-variance early training regime. Signal from optimizer differences is dominated by noise from batch sampling, initialization effects, and learning rate sensitivity. A single-seed comparison at 20M tokens is unlikely to produce a statistically significant result. The defender acknowledges this ("noisier signal") but dismisses it. A null result here is ambiguous — does HeadOrtho not work, or did we not train long enough? This makes the 24-hour conclusion fundamentally fragile.

---

## Deep Evaluation

**On R1 vs D1:** The mode-2 fix is technically sound, but it papers over a deeper question that D1 raises. Teon's insight is that *correlated* gradients benefit from joint orthogonalization. If head gradients are uncorrelated (designed to specialize), then joint orthogonalization adds nothing — you're just doing Muon on a reshaped matrix that happens to be square. The mode-2 fix solves the *numerical* problem but doesn't address the *statistical* one: are these gradients actually correlated enough to benefit?

However, the defender has a valid escape hatch: the Step 0 diagnostic directly tests this. If alignment is measurable, D1 is refuted by data within 30 minutes. The burden of proof shifts cleanly.

**On R2 vs D2:** The "never worse" guarantee is real but potentially trivial. If tensor ortho across uncorrelated heads is equivalent to standard Muon up to rotation, then you've built a more complex optimizer that produces the same result. The research contribution is zero even if it "doesn't hurt." A positive result requires actual improvement, not just non-degradation.

But: the defender isn't claiming the impossibility result. They're claiming that empirical alignment *might* exist (it's untested), and if it does, the gains are real. This is a legitimate exploration. The question is whether a single 20M token run produces enough signal to tell the difference.

**On R3 vs D3:** This is the crux. The revised plan is lean, but the fundamental signal-to-noise problem at 20M tokens is real. Teon's improvements at 10B tokens are ~2% PPL reduction. At 20M tokens with a single seed, natural variance between runs is easily 1-3% PPL. You might get a positive result that's just noise, or a null result that hides a real effect.

Counter-point: the defender's prior work showed clear, unambiguous differences between QK-Norm and No-QK-Norm at 20M tokens on this exact model. So the setup *can* distinguish optimizer variants — the question is whether HeadOrtho's effect is as large as the QK-Norm effect. If it is, 20M tokens suffices. If it's a subtle 0.5% improvement, it won't be visible.

**The deciding factor:** Is this a project where the downside is bounded and the upside is a genuinely novel finding?

Yes. The worst case is: you spend 10 hours of GPU time, get a null result, and write "HeadOrtho did not improve val loss at 20M tokens; longer training may be needed." That's a minor loss. The best case is: you discover that cross-head tensor orthogonalization measurably improves training, which is a novel result that directly extends Teon to a new axis (heads instead of layers). That's publishable.

The Step 0 diagnostic is the key gate. If it shows no cross-head alignment, the project pivots to a simpler "per-head Muon" study or gets dropped immediately — you lose 30 minutes, not 24 hours.

---

## VERDICT: REVISE

**The Deciding Factor:** The mode-2 solution transforms the weakest technical objection into the project's strongest asset (perfect square matrix), and the Step 0 diagnostic creates a fast-fail gate that prevents wasting GPU time on a dead idea. The bounded downside ("never worse than Muon") + fast-fail gate + lean revised plan = acceptable risk for a 24-hour project.

**Executive Summary:** The critique correctly identified that the original plan was unfocused, used a numerically unstable matrix configuration, and tried to do too much in 24 hours. The defense successfully resolved all three issues: mode-2 gives a square matrix, the plan was cut to 2 experiments with a single metric, and a 30-minute diagnostic was added as a go/no-go gate. The remaining risk — that 20M tokens may not show a significant result — is acceptable because the prior work on this exact setup has shown that optimizer differences are visible at this scale.

---

## Mandatory Blueprint (V1.1 Final)

### Step 0: Diagnostic (30 min) — HARD GATE
Run cross-head singular vector alignment check on Q/K gradients.
- If mean cosine > 0.3 for left singular vectors → **GO with mode-2**
- If mean cosine > 0.3 for right singular vectors → **GO with mode-1**
- If neither > 0.3 → **DROP the project. Do not proceed.**

### Step 1: Code Update (30 min)
- Update `HeadOrthoMuon` default `ortho_mode` based on Step 0
- Run 10-step smoke test: verify no crash, measure per-step overhead

### Step 2: Main Experiment (4–8 hrs)
- Single run: HeadOrtho Muon on 1.5B model, 20M tokens (or 8M if GPU-constrained)
- Baseline: reuse existing Muon results from `research_results/`

### Step 3: Result (1 hr)
- Primary metric: **Val loss** (HeadOrtho vs baseline)
- Secondary: head diversity cosine similarity at final checkpoint (optional)
- Generate val loss comparison plot

### Step 4: Write-up (2 hrs)
- One-page result: setup, val loss curve, one conclusion sentence
- If positive: "HeadOrtho improves val loss by X% over Muon at 20M tokens"
- If null: "HeadOrtho shows no significant difference at 20M tokens; longer training needed"
- Post to social media

### Non-Negotiable Conditions
1. Step 0 diagnostic **must** be done before any GPU training
2. Mode-2 is the default unless Step 0 data says otherwise
3. Only 2 experiments (baseline reuse + HeadOrtho)
4. Single seed is acceptable for a 24-hour result but must be disclosed
