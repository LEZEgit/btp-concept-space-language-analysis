## Experimental Methodology & Implementation Guide: Investigating Principal Latent Languages in Sarvam-2B

### 1. Research Objective & Conceptual Overview

In *Do Llamas Work in English? On the Latent Language of Multilingual Transformers* (Wendler et al., 2024), the authors use the **Logit Lens** technique to probe intermediate hidden states of Llama-2 during non-English translation tasks. Their core finding is that transformers process information through three phases—**Input Space**, **Concept Space**, and **Output Space**—where the intermediate "Concept Space" (Phase 2) disproportionately assigns probability mass to **English** tokens before rotating into the target language at the final layers.

However, Llama-2 was trained on an 89.7% English-dominated corpus. **Our experiment** tests whether this English-pivot hypothesis still holds for an **Indic-centric foundation model (`sarvamai/sarvam-2b-v0.5`)** when translating directly between two non-Hindi Indian regional languages (e.g., **Bengali $\rightarrow$ Tamil**), or whether **Hindi** (`hi`) or the **source regional language** (`input_lang`) acts as a competing pivot in the concept space.

To test this, we expand the original two-language tracking (`en` vs. `target_lang`) into a **4-candidate principal language framework**:


$$\text{Candidates} = \{\text{English } (\texttt{en}), \text{ Hindi } (\texttt{hi}), \text{ Source } (\texttt{input\_lang}), \text{ Target } (\texttt{target\_lang})\}$$

---

### 2. Summary of Similarities vs. Custom Modifications

| Pipeline Component | Original Paper (`Translation.ipynb`) | Our Implementation (Sarvam-2B Multi-Candidate) |
| --- | --- | --- |
| **Base Model & Architecture** | `meta-llama/Llama-2-7b-hf` (32 layers, $d=4096$, $v=32,000$)

 | `sarvamai/sarvam-2b-v0.5` (28 layers, $d=2048$, $v=64,128$)

 |
| **Data Merging** | Merges 2 CSVs (`input_lang`, `target_lang`) on English

 | Merges 3+ CSVs dynamically across all candidates (`en`, `hi`, `ben`, `tam`)

 |
| **Token Prefix Matching** | SentencePiece `▁` prefix + `<0x..>` byte fallback for `zh`/`ru`

 | SentencePiece `▁` prefix + `tokenizer.encode()` fallback for Indic ligatures

 |
| **Latent Language Tracking** | Tracks only 1 latent language (`lang_latent='en'`) + target

 | Simultaneously tracks all 4 candidate languages in `lang_probs` dictionary

 |
| **VRAM & Device Handling** | Unembeds entire sequence `[L, seq_len, d]`; CPU/GPU mismatch bug

 | Patched `get_logits` device; slices last token `[L, d]` *before* `unemb` (saves ~95% VRAM)

 |
| **Visualization** | 1 plot (`en` vs `target_lang`) + 1 energy plot

 | $2 \times 2$ comparative grid + 4 individual paper-style entropy heatmap PDFs + energy plot

 |

---

### 3. Block-by-Block Technical Explanation

#### **Block 1: Environment, Imports & Hardware Pinning** (Cell 4)

* **What it does:** Loads PyTorch, Matplotlib, `LlamaHelper` (from `llamawrapper.py`), and plotting utilities (`plot_ci`, `plot_ci_plus_heatmap` from `utils.py`).


* **What stayed the same:** Standard imports and fixed random seeds (`seed = 42` on lines 21–23) for reproducible few-shot sampling.


* **What we modified:**
* Added `sys.path.append('/raid/home/dgx1405/btp1-2026/llm-latent-language')` (line 7) to resolve local module imports.


* Added `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` (line 19) after querying `!nvidia-smi` (Cell 2) to isolate the process onto a single free GPU on the shared DGX server and prevent multi-GPU fragmentation.





---

#### **Block 2 & Block 3: Experiment Parameters & Candidate Array Setup** (Cells 5 & 6)

* **What it does:** Defines the source/target translation pair, model identifier, token-filtering flags, and the list of candidate pivot languages.


* **What stayed the same:** Mutually exclusive validation check `if single_token_only and multi_token_only` in Block 3.


* **What we modified (Block 2):**
* Set `input_lang = 'ben'`, `target_lang = 'tam'`, `model_size = '2b'`, and `custom_model = 'sarvamai/sarvam-2b-v0.5'` (lines 1–4).


* Defined `principal_lang_candidates = ["en", "hi", input_lang, target_lang]` (line 10) and deduplicated it while preserving order via `candidates = list(dict.fromkeys(principal_lang_candidates))` (line 12) so the pipeline still works cleanly if `input_lang` or `target_lang` is ever set to `'en'` or `'hi'`.





---

#### **Block 4: Model Initialization, Unembedding Extraction & Energy Baseline** (Cell 7)

* **What it does:** Loads Sarvam-2B in 8-bit quantization, constructs the Logit Lens unembedding operator, and precomputes the normalizing denominator for the token energy metric.


* **What stayed the same:**
* Constructing `unemb = nn.Sequential(llama.model.model.norm, llama.model.lm_head)` (line 5), which applies the final RMSNorm layer followed by the vocabulary projection matrix $U \in \mathbb{R}^{64128 \times 2048}$.


* Computing the row-normalized weighted unembedding matrix $\hat{U}$ (`U_normalized`, lines 11–13) and the root-mean-square cosine similarity among all vocabulary tokens (`avgUU`, lines 15–16), corresponding to the denominator $\frac{1}{v}\Vert{}\hat{U}\hat{U}^\top\Vert{}_F = \frac{1}{v}\Vert{}\hat{U}^\top\hat{U}\Vert{}_F$ in Equation (2) of the paper ($0.04994$ for Sarvam-2B).




* **What we modified:** Removed the redundant `load_unemb_only(model_size)` call from Cell 4 of the original code, which hardcoded local EPFL server paths (`/dlabdata1/llama2_hf/...`).



---

#### **Block 5: Multi-Language Dataset Loading & Alignment** (Cell 8)

* **What it does:** Loads the cleaned word-translation datasets (`132` concept nouns), checks single-token coverage in Sarvam-2B's vocabulary (`70 of 132` for Tamil), and inner-joins all 4 candidate languages into a unified DataFrame `df_merged`.


* **What stayed the same:**
* Vocabulary single-token counting (`if word_str in tokenizer.get_vocab() or '▁' + word_str in tokenizer.get_vocab():`, lines 6–15).


* Substring filtering loop (`if str(row['en']).lower() in str(row[target_lang]).lower():`, lines 28–31) to remove loan words/cognates that contain the literal English string.




* **What we modified:**
* The original code (Cell 5 & Cell 9) only loaded two CSVs (`df_en_fr` and `df_en_de`) and merged them once.


* In lines 18–25, we start with `df_tgt` (`en` and `tam`) and dynamically loop over `for lang in candidates:` to merge `hi/clean.csv` and `ben/clean.csv` on `['en']`, producing a 132-row aligned DataFrame with columns `['en', 'tam', 'hi', 'ben']`.





---

#### **Block 6: Token Prefix & Indic Vocabulary Lookup Helpers** (Cell 9)

* **What it does:** Implements Appendix A.2 of the paper—accounting for multi-token words and leading whitespace variations when measuring the probability of a target word in a given language.


* **What stayed the same:**
* `token_prefixes(token_str)` (lines 1–5) generates all character substrings $t_{1:i}$ from length $1$ to $n$.


* `add_spaces(tokens)` (lines 7–8) prepends the SentencePiece whitespace character `'▁'` to each prefix.


* `process_tokens(...)` (lines 22–30) looks up every prefix variant in `tokenizer.get_vocab()` and collects their token IDs.




* **What we modified:**
* Added `.strip()` in `token_prefixes` (line 2) to sanitize trailing whitespace from CSVs.


* Added an Indic ligature fallback (lines 31–35): `if len(final_tokens) == 0: encoded = tokenizer.encode(str(token_str).strip(), add_special_tokens=False)`. In Indic scripts (Devanagari, Bengali, Tamil, Telugu), Unicode combining vowel signs (*matras*) or conjunct consonants (*yuktakshar*) can cause character-by-character slicing to miss multi-character subword tokens; falling back to `encoded[0]` guarantees every word yields at least its true leading subword token ID.





---

#### **Block 7 & Block 8: Few-Shot Prompt Construction & Multi-Candidate Token ID Extraction** (Cells 10 & 11)

* **What it does:** Follows Section 3.3 (*Translation task*) of the paper to construct 5-shot in-context translation prompts of the form:

$$\texttt{বাংলা: "দশ" - தமிழ்: "பத்து"}\dots\texttt{বাংলা: "বই" - தமிழ்: "}$$



and records the valid token ID sets for all 4 candidate languages.


* **What stayed the same:**
* `compute_entropy(probas)` (lines 6–7 of Block 7) computes Shannon entropy $H(P) = -\sum_x P(x)\log_2 P(x)$ over the vocabulary distribution at each layer (added `+ 1e-12` inside `torch.log2` to prevent `NaN` on zero-probability tokens).


* Sampling $k-1=4$ random context pairs from `df[df.index != ind]` and appending the $k$-th target test word at the end (lines 31–39 of Block 7).




* **What we modified:**
* Expanded `lang2name` (lines 10–21) with native script names (`বাংলা`, `தமிழ்`, `हिन्दी`, etc.) for prompt construction, and added `lang2english_name` (lines 22–29) with ASCII names (`Bengali`, `Tamil`, `Hindi`) so Matplotlib titles render without missing-glyph warnings.


* Replaced the single `latent_token_id` extraction with a loop over all 4 candidates (`for c in candidates:`, lines 52–66 of Block 7), storing `en_token_id`, `hi_token_id`, `ben_token_id`, and `tam_token_id` in each dataset record (Block 8, Cell 11 output table).


* Made the prefix-intersection filter configurable via `strict_disjoint=False` (lines 60–63) so shared Indic script prefixes or `c == target_lang` do not erroneously discard valid samples.





---

#### **Block 9: Device Patch, Memory-Optimized Logit Lens & Energy Calculation** (Cells 13 & 14)

* **What it does:** Executes the forward pass on all 132 prompts, applies the Logit Lens (`unemb`) to the final token's hidden state $h_n^{(j)}$ at every layer $j \in \{1, \dots, 28\}$, sums probability mass over each candidate language's token set, and computes layer-wise entropy and squared token energy.


* **What stayed the same:**
* Probability summation formula from Section 3.4: $P(\text{lang} = \ell \mid h_n^{(j)}) = \sum_{t \in \mathcal{T}_\ell} P(x_{n+1} = t \mid h_n^{(j)})$ via `last[:, tok_ids].sum(dim=-1)` (Cell 14, line 23).


* Geometric token energy formula from Section 4.2, Equation (2): normalizing $h_n^{(j)}$ (`latents_normalized`, lines 31–32) and computing $E(h) = \frac{\text{RMS}(\hat{U} h)}{\text{avgUU}}$ (`norm / avgUU`, lines 33–34).




* **What we modified:**
* **Device Mismatch Patch (Cell 13, lines 2–10):** `LlamaHelper.get_logits` in `llamawrapper.py` left tokenized `input_ids` on CPU while `embed_tokens` resided on `cuda:0`. We monkey-patched `llama.get_logits` with `.to(embed_device)`.


* **95% VRAM Reduction (Cell 14, lines 8–19):** The original notebook ran `logits = unemb(latents)` on the full sequence tensor `[28, seq_len, 2048]`, allocating a massive `[28, seq_len, 64128]` tensor on GPU before slicing `logits[:, -1, :]`. We reversed the order—slicing `last_latents = latents[:, -1, :]` (`[28, 2048]`) *first*, deleting `latents`, and projecting only the final token position inside a `with torch.no_grad():` context, reducing the entire 132-sample loop to 2 minutes 16 seconds with zero OOM errors.





---

#### **Block 10 & Block 11: Multi-Candidate Visualization & Artifact Serialization** (Cells 15 & 16)

* **What it does:** Visualizes the layer-by-layer probability trajectories (with 95% bootstrap confidence intervals via `plot_ci`), entropy heatmaps (`plot_ci_plus_heatmap`), and token energy curves, and saves the raw tensors to disk.


* **What stayed the same:**
* Individual paper-style plots (Block 10, lines 38–56) calling `plot_ci_plus_heatmap` with the target language (`tam`, blue curve) overlaid via `plot_ci(ax2, lang_probs[target_lang], ...)`.


* Layer-wise token energy plot (`plot_ci(ax2, energy, 'energy', color='tab:green', ...)` in Block 11, lines 1–9).




* **What we modified:**
* Added `'2b': 4` to `size2tik` (Block 10, line 1) for 28-layer x-axis tick spacing.


* Created a unified **$2 \times 2$ comparative grid** (Block 10, lines 13–36) plotting each candidate language (`en` in orange, `hi` in green, `ben` in purple, `tam` in blue) in its own subplot against the `tam` target curve for side-by-side comparison.


* Saved both `latents.pt` (`torch.Size([132, 28, 2048])`) and the full `lang_probs.pt` dictionary (Block 11, lines 11–12) for downstream cross-language analysis.
