# Primary-source baseline for a small Wordle policy

Research cut: 2026-08-23. This note separates source facts from planning inferences. It is a technical baseline, not a legal opinion.

Decision update, 2026-08-23: the approved design supersedes the recommendations below. The project starts from a purpose-built, randomly initialized DeepSets policy and keeps pretrained language models as optional comparison runs. It equally supersedes the corpus recipe below: the frozen benchmark builds its 5,201 accepted guesses from SCOWL v1 `rel-2020.12.07` American size 60 words and its primary answer profile from the size 35 intersection with WordNet 3.0 five-letter lemmas, pinned at commit `5ef55f9c42730ebe4394a78b77855468a6f15dd2`, rather than the SCOWL v2 size-70 list at commit `1e5b7d3a72f47a71da5d28686c1dd4b397178485`. The source, licensing, rules, and competitor evidence remain part of the research record.

## Decision

Start with **SmolLM2-360M Base**, a 360M-parameter decoder released under Apache 2.0, and distill an exact Wordle solver into a finite-action policy. Its official model card reports a 723.56 MB BF16 footprint, so a LoRA pilot is the sensible first M4 experiment; full-model training is possible only after measuring activation memory and throughput on the actual machine. [SmolLM2-360M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)

Do not start at 2B. Wordle's state and action spaces are small enough to solve symbolically, and a 2.9M-parameter purpose-built network already reports 99.56% wins without a dictionary or search at inference. A larger language model must beat that strict neural baseline and approach the proven symbolic optimum to justify its footprint. [Wordle Neural Solver repository](https://github.com/talontownsend/wordle-neural-solver)

The headline target must be qualified. On the frozen original 2,315-answer benchmark, the expected-score optimum is already proved: 7,920 total guesses, or 3.421166307 per answer, in normal mode. A neural policy cannot beat that lower bound under the same rules and uniform prior; it can only match it. [Cunanan and Thielscher, IJCAI 2023](https://www.ijcai.org/proceedings/2023/0615.pdf)

## Game contract

### Source facts

The current official game says the player has six tries to guess a hidden five-letter word, with a new puzzle daily. The official game is the canonical product surface. [NYT Wordle](https://www.nytimes.com/games/wordle/index.html)

The current settings bundle describes hard mode as requiring revealed hints in later guesses and permits enabling it only at the start of a round. [NYT settings bundle, inspected 2026-08-23](https://www.nytimes.com/games-assets/v2/423.53a79c03e7a1f689b1b3.js)

The current validator fixes each prior green in the same position and requires a later guess to contain at least the revealed multiplicity of every green or yellow letter. It does not require the new guess to be consistent with every prior gray, and a yellow can be reused in its old position. This is weaker than “guess only from the remaining answer set.” [NYT validator bundle, inspected 2026-08-23](https://www.nytimes.com/games-assets/v2/8417.ebe1fe433c7d632b1e0f.js)

Duplicate-letter feedback consumes hidden-word multiplicity: after greens are assigned, only as many remaining copies can be yellow as occur in the answer; excess copies are gray. The exact-solver paper formalizes this feedback function. [Bertsimas and Paskov preprint](https://nickpainter.github.io/optimal-wordle/assets/Wordle_Paper_Final.pdf)

The JS asset URLs are versioned and may move. Archive their hashes and write executable conformance cases now, including `allee/eagle`-style duplicate cases and cases showing the difference between hard legality and full candidate consistency.

### Planning contract

Implement one deterministic engine shared by data generation, training checks, and evaluation. Version and hash all of the following together:

1. answer set `S`, accepted-guess set `G`, and answer prior;
2. duplicate-aware feedback function;
3. normal and current-NYT hard validators;
4. six-turn termination and the score assigned to a loss;
5. tie-breaking order for equally valued actions.

Keep three inference classes separate because they answer different scientific questions:

- **Board-only neural:** the network sees guesses and colors and emits a word; no dictionary, candidate filtering, legality mask, search, or fallback. This is the direct comparison with the 2.9M neural solver.
- **Symbolic-state policy:** a rules engine computes exact board state and hard legality, then a learned policy chooses an action without lookahead. This is the recommended product path.
- **Search-guided system:** the policy or value model orders a bounded symbolic search. This may be strongest in operation, but it is not a model-only result.

## Frozen benchmarks and known ceilings

| Benchmark | Primary-source result | Consequence |
|---|---|---|
| Original normal mode, `|G| = 12,972`, `|S| = 2,315`, uniform answers | Cunanan and Thielscher prove that the unique surviving first guess is `SALET` and that a complete strategy reaches total 7,920, mean 3.421166307. [IJCAI paper](https://www.ijcai.org/proceedings/2023/0615.pdf) Alex Selby's independent certificate file also records `salet 7920`. [normal certificate](https://github.com/alex1770/wordle/blob/main/normal.top105.proven) | This is the non-negotiable classic normal target. Equaling 7,920 is “optimal on this frozen benchmark,” not “better than Wordle.” |
| Original hard mode, same 2,315 answers | Selby's hard-mode certificate records `salet 8122`, mean 3.508423326. [hard certificate](https://github.com/alex1770/wordle/blob/main/hard.top101.proven) | Reproduce 8,122 with our current hard validator before treating the number as comparable; any validator mismatch creates a new benchmark. |
| Alternate exact formulation | Bertsimas and Paskov use 2,315 states but 10,657 actions, formulate exact dynamic programming, and report `SALET`, mean 3.421, maximum five in normal mode. Their published paper also studies hard mode, nonuniform priors, and worst-case objectives. [Operations Research article](https://pubsonline.informs.org/doi/full/10.1287/opre.2022.0434), [author preprint](https://nickpainter.github.io/optimal-wordle/assets/Wordle_Paper_Final.pdf) | Do not combine results across papers unless the exact action list is named. The same score does not erase the 10,657-versus-12,972 list difference. |
| Near-optimal online rollout | Bertsekas reports 3.4345 normal versus 3.4212 exact, and 3.5231 hard versus 3.5084 exact, with decisions computed in seconds. [original paper](https://web.mit.edu/dimitrib/www/Playing_Wordle_Using_an_Online_Rollout_Algorithm_for_Deterministic_POMDPs.pdf) | A student should beat this practical-search baseline before “near optimal” means much. |
| Strict neural competitor, 14,855 words used as both guesses and secrets | The author reports exhaustive deterministic evaluation of a 2.9M-parameter board-only model: 99.56% wins and 4.24 guesses among wins in normal mode; 89.81%, 4.31, and zero violations over 66,580 guesses in hard mode. It has no dictionary, mask, search, or fallback at play time. [repository and evaluation](https://github.com/talontownsend/wordle-neural-solver) | These figures are author-reported and use a much harder but different answer distribution. Re-run the repository at a pinned commit before quoting a comparison. |

Selby's exact implementation and strategy certificates are MIT-licensed software. Its README documents original and later reconstructed Wordle list modes. Treat those lists as historical benchmark artifacts, not as a first-party NYT data license. [solver repository](https://github.com/alex1770/wordle), [MIT license](https://github.com/alex1770/wordle/blob/main/LICENSE)

## Word lists that can be published with the project

No current first-party NYT page found in this research supplies a downloadable, redistributable complete answer-and-guess corpus under an explicit data license. The safe project structure is to keep the historical Wordle benchmark reproducible without republishing its lists, while making the public training and held-out corpus from sources with explicit terms.

| Source | Primary terms | Recommended use |
|---|---|---|
| SCOWL / English Speller Database v2 | The upstream copyright file permits use, copying, modification, distribution, and sale subject to retaining the listed notices. Additional notices attach to some source dictionaries and generated data, so the exact configuration matters. [upstream repository](https://github.com/en-wl/wordlist/tree/v2), [copyright file](https://github.com/en-wl/wordlist/blob/v2/Copyright) | Pin commit `1e5b7d3a72f47a71da5d28686c1dd4b397178485`, build the American size-70 spelling list, retain notices, and record every filter. This is the main redistributable `G_open`. |
| Princeton WordNet 3.0 | Princeton permits use, copying, modification, and distribution without fee if its copyright notice and warranty disclaimer remain. [official license](https://wordnet.princeton.edu/license-and-commercial-use), [official 3.0 download](https://wordnetcode.princeton.edu/3.0/) | Use single-token lowercase five-letter lemmas as one auditable signal for `S_open`, not as a claim of commonness. |
| `wordfreq` | The official README licenses code under Apache 2.0 and its combined data under CC BY-SA 4.0, with source-specific attribution requirements. [license section](https://github.com/rspeer/wordfreq/blob/master/README.md#license) | Pin commit `912caf64b657478d1dff1138efdc078947d54bb1`; use frequency only as a published prior or answer-set filter when the derivative-data and attribution obligations are acceptable. |

Build two open evaluation regimes from those sources. In `open-uniform`, answers are sampled uniformly from the declared `S_open`. In `open-frequency`, freeze normalized `wordfreq` weights. Generate `G_open` and `S_open` with a checked-in script, emit sorted files with SHA-256 digests, retain all notices, and publish the counts only after generation. Split answer words into folds by `SHA256(word + split_salt)`, not by random row order, and never let held-out fold answers appear in oracle trajectories, synthetic games, prompts, or hyperparameter selection.

This open benchmark measures transfer to a reproducible English-word game. It is not the current NYT editorial distribution, and it should never be labeled “official Wordle.”

## Checkpoint choice

| Checkpoint | Source facts | Decision |
|---|---|---|
| [SmolLM2-360M Base](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | 360M decoder, Apache 2.0, 8,192-token context, 723.56 MB reported BF16 footprint. Its official config has 32 layers, width 960, and a 49,152-token vocabulary. [config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json) | **Primary local baseline.** It is the cleanest Apache-licensed checkpoint near the requested 0.2–0.4B range. Start from Base because the task is policy learning, not chat imitation. |
| [SmolLM2-135M Base](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 135M decoder, Apache 2.0, 8K family context. | Capacity floor. If it matches 360M, ship the smaller model. |
| [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m) | Google describes a 270M text model with 32K context. Access requires accepting the custom Gemma license. [official model card](https://ai.google.dev/gemma/docs/core/model_card_3), [terms](https://ai.google.dev/gemma/terms) | Closest nominal size, but use only as a controlled comparison after accepting the redistribution and derivative-model terms. It is not Apache-licensed. |
| [Qwen3-1.7B Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | 1.7B text decoder, 1.4B non-embedding parameters, 28 layers, 32,768-token context, Apache 2.0. | Preferred large ablation because it is text-only and permissively licensed. |
| [SmolLM2-1.7B Base](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 1.7B decoder, Apache 2.0; official BF16 example reports 3,422.76 MB. | Same-family scale ablation, which isolates capacity better than changing families. |
| [Qwen3.5-2B Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | Apache 2.0; the card describes a 2B language model plus a vision encoder, 24 hybrid layers, 248,320 output vocabulary, and 262,144 native context. | Exact nominal 2B option, but its vision and vocabulary overhead buy nothing for Wordle. Do not use first. |

Wordle history is far below every listed context limit, so context length does not decide this choice. License clarity, local training cost, and policy-head fit do.

## Training design

The teacher is an exact search oracle, not another language model. Represent a state by guess/color history plus the exact remaining-answer bitset. For each legal action, compute expected remaining cost and retain the optimal action, cost-to-go, and regret above optimum. Exact dynamic programming is established for this game, although Bertsimas and Paskov report that their full original solve took days on a 64-core machine; cache every solved state by benchmark hash. [Bertsimas and Paskov preprint](https://nickpainter.github.io/optimal-wordle/assets/Wordle_Paper_Final.pdf)

Attach a dedicated action scorer over the frozen guess vocabulary instead of asking the language-model softmax to spell five letters. Apply the game legality mask after scoring. This guarantees accepted words for the symbolic-state track and avoids wasting most of a 49K-token vocabulary on an action space of at most roughly 15K words. Keep the board-only letter decoder as a separate strict ablation.

Train in four stages:

1. **Reproduce the oracle.** Exhaustively recover 7,920 normal and 8,122 hard on the frozen original benchmark. Stop if either total differs; data from a mismatched engine would only teach the mismatch.
2. **Behavior cloning.** Supervise the policy on oracle actions and the value head on exact cost-to-go. Use a temperature-scaled distribution over negative action regret when full action values are available. Classical knowledge distillation transfers a teacher distribution, and policy distillation applies that idea to policies. [Hinton, Vinyals, and Dean](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/), [Rusu et al.](https://arxiv.org/abs/1511.06295)
3. **DAgger recovery.** Roll out the student, query the oracle on states the student actually visits, aggregate those examples, and repeat. This addresses compounding error outside the oracle's clean trajectory distribution. [Ross, Gordon, and Bagnell](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
4. **Open-list curriculum.** Generate licensed game families with different answer subsets, priors, normal/hard rules, and trap-heavy word clusters. Keep evaluation folds unseen. Mix classic benchmark states only into the classic head or adapter so memorizing 2,315 answers cannot masquerade as generalization.

Use policy cross-entropy or KL loss plus a value regression loss. Report ablations for value loss, regret-soft targets, DAgger, candidate-bitset input, hard mask, and base-model scale. Skip chain-of-thought targets: the deployable action is a five-letter choice, while exact costs provide a cleaner training signal than prose rationales.

## Memory and local-compute envelope

Hugging Face's framework documentation budgets mixed-precision full Adam training at 6 bytes per parameter for FP16 plus FP32 weights, 8 for Adam states, and 4 for gradients, or about 18 bytes per trainable parameter before activations and temporary buffers. It also notes that activations depend on batch, sequence length, depth, and width. [Transformers memory anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy)

| Model size | BF16 weights, decimal | Full Adam static tensors, decimal | Interpretation |
|---:|---:|---:|---|
| 135M | 0.27 GB | 2.43 GB | Cheap capacity floor. |
| 360M | 0.72 GB | 6.48 GB | Plausible M4 full-tune only after an actual one-batch memory test; LoRA is the safer first run. |
| 1.7B | 3.40 GB | 30.60 GB | Full Adam is outside a typical 16–24 GB local budget before activations. |
| 2.0B language model | 4.00 GB | 36.00 GB | Use LoRA/QLoRA locally or move full fine-tuning to larger/sharded hardware. |

These are arithmetic estimates from the framework's bytes-per-parameter accounting, not measured peak memory. LoRA freezes the base and trains low-rank adapters; QLoRA combines adapters with a frozen 4-bit base. [LoRA paper](https://arxiv.org/abs/2106.09685), [QLoRA paper](https://arxiv.org/abs/2305.14314), [Hugging Face quantized-PEFT guide](https://huggingface.co/docs/peft/main/developer_guides/quantization)

For the first M4 run: use SmolLM2-360M Base, sequence length 128 or less, LoRA on attention and MLP projections, gradient accumulation, and a single-batch dry run that records peak unified memory and states per second. Promote to full fine-tuning only if the measured headroom is comfortable. Promote to 1.7B only if 360M retains measurable oracle regret after DAgger and data scaling.

## Claim protocol

A defensible public result needs three leaderboards rather than one blended score.

### Classic exact leaderboard

Run every one of the 2,315 answers deterministically. Publish total guesses, mean, maximum, solve rate within six, hard-rule violations, mean oracle regret, p95 latency, peak memory, model hash, list hashes, and whether symbolic candidate computation or search was available. A result of exactly 7,920 normal or 8,122 hard may be called optimal only on the named frozen rules, lists, prior, and inference class. Include the complete decision tree or a replay log so others can verify every game.

### Open held-out leaderboard

Publish the licensed corpus build and fixed hash folds. Report macro results across folds and paired per-answer differences against the exact oracle, rollout solver, 2.9M neural baseline, 135M, 360M, and 1.7B policies. Hyperparameters and stopping decisions use development folds only. A claim here should read “best tested board-only policy on OpenWordle-v1,” or the corresponding symbolic-state class, never “best Wordle player.”

### Forward NYT ledger

Freeze the model, code, accepted-guess behavior, and benchmark statement before a consecutive daily run. Record the answer only after the day's game closes; the NYT exposes dated puzzle data through its service, but using the current date during play would leak the answer. [NYT dated Wordle service pattern](https://www.nytimes.com/svc/wordle/v2/2026-01-01.json)

Pre-register at least the run length, missed-day policy, normal versus hard mode, network/search access, latency budget, invalid-guess handling, and loss scoring. Compare paired results over the same days and give uncertainty intervals. This can support “best measured over this preregistered period among these systems.” It cannot prove an unqualified world-best claim because the future editorial distribution and the universe of private solvers are unknown.

## Release gates

1. Engine tests match duplicate feedback and current hard behavior, then exact replay matches 7,920 and 8,122.
2. Every training and evaluation list has provenance, commit, license notices, build command, and SHA-256 digest.
3. SmolLM2-360M beats the 2.9M board-only competitor under the same 14,855-secret protocol or clearly wins a stricter, predeclared protocol.
4. The symbolic-state 360M policy reaches zero or near-zero oracle regret on classic and beats rollout on held-out open folds.
5. The 1.7B run happens only if the 360M learning curve shows a capacity ceiling; 2B is not a milestone by itself.
6. The release names its inference class everywhere and never compares model-only, masked, and search-assisted scores as if they were the same game.

The strongest attainable statement is mathematical and narrow: **optimal on a frozen benchmark, under a named inference contract**. The strongest live statement is empirical and time-bounded. Keeping those distinct is what makes “best” credible.
