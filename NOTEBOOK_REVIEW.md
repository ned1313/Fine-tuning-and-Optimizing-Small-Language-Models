# Notebook Markdown Review

Review date: 2026-09-08. All proposals are pending approval. No notebooks have been changed as part of this review.

## Approval Workflow

Change each finding's **Decision** from `Pending` to `Approved`, `Rejected`, `Fixed Manually`, or `Revise`, and edit the suggested text or add approval notes as needed. Approval applies only to the described Markdown edits. Any accompanying code change requires explicit approval and is identified separately.

Locations use 1-based notebook cell numbers from this review snapshot, plus headings or code symbols to identify the content after cells move. Suggested file paths inside replacement prose are relative to the repository root unless stated otherwise.

Priorities: **High** means instructions can block a workflow or materially mislead it; **Medium** means a factual or interpretive error; **Low** means consistency, navigation, or editorial polish.

## Findings

### NB-01: GRPO Input Locations Do Not Match the Repository

**Priority:** High. **Decision:** Fixed Manually. **Scope:** Markdown with a separately identified code dependency.

**Location:** [notebooks/grpo_lora_run.ipynb](notebooks/grpo_lora_run.ipynb), Cell 1, "What you need"; Cells 5 and 11 provide the implementation evidence.

**Finding:** The introduction says the database and JSONL files are shipped alongside the notebook under `data/`. They actually live in [datasets/data](datasets/data). The code also resolves the database and splits directly under `datasets/`, so correcting only the introduction will not make the notebook run.

**Suggested edit:** Replace the Files row with: "SQL inputs are in the repository's `datasets/data/` directory: `taco_alley.db`, `train.jsonl`, `eval_params.jsonl`, and `eval_templates.jsonl`. Ensure `DATASETS_DIR` points to that directory before running the database and dataset cells."

**Separate code approval needed:** Set `DATASETS_DIR = REPO_ROOT / "datasets" / "data"`. Do not silently fold this change into Markdown approval.

### NB-02: Colab Credential Instructions Target the Wrong Process

**Priority:** High. **Decision:** Fixed Manually. **Scope:** Markdown; an automated Colab Secrets integration would be a separate code change.

**Location:** [notebooks/basic_sft_example.ipynb](notebooks/basic_sft_example.ipynb), Cell 20, "Upload the trained model"; [notebooks/data_generation.ipynb](notebooks/data_generation.ipynb), Cell 1, "API key setup", and Cell 5, "Validate local secrets file".

**Finding:** Both notebooks instruct learners to set an environment variable in the Colab terminal. An export in a separate shell does not populate the already-running notebook kernel's environment. Basic SFT reads `HF_TOKEN`; data generation reads `OPENAI_TOKEN`. Data generation's Cell 6 loads a local secrets file if available but does not require one when the environment variable is already present, contrary to the mandatory wording in Cell 5.

**Suggested edit:** "For local runs, keep your credential in a `.env` file in the notebook's working directory. In Colab, use Colab Secrets and explicitly load the value into the notebook kernel's environment before the client or upload cell. Creating a secret alone does not set an environment variable. Use `HF_TOKEN` for Hugging Face or `OPENAI_TOKEN` for this notebook's OpenAI client, as applicable. Do not paste credentials into saved cells or outputs."

For data generation Cell 5, use the heading "Load local credentials (optional)" and: "The next cell attempts to load `.env` from the working directory. It is optional when `OPENAI_TOKEN` is already available in the kernel environment."

**Approval note:** A ready-to-run Colab Secrets helper would improve usability but requires separate approval to add a code cell. Preserve the repository's `OPENAI_TOKEN` name unless the code is also deliberately changed.

### NB-07: Colab Prep Omits Repository Files and Working Directory

**Priority:** High. **Decision:** Approved. **Scope:** Markdown only, except the separate GRPO path fix in NB-01.

**Location:** Cell 3 in all eight notebooks, "Google Colab prep". Path evidence: [notebooks/basic_sft_example.ipynb](notebooks/basic_sft_example.ipynb), Cell 16; [notebooks/data_generation.ipynb](notebooks/data_generation.ipynb), Cell 9; [notebooks/datasets.ipynb](notebooks/datasets.ipynb), Cell 18; [notebooks/grpo_lora_run.ipynb](notebooks/grpo_lora_run.ipynb), Cell 5. The three complaint SFT notebooks and DPO also consume repository files.

**Finding:** The prep instructions cover packages but not repository data or the kernel working directory. Opening a notebook through its Colab badge does not provision the whole repository in the runtime. Four notebooks calculate the repository root as the parent of the current working directory. Consequently, data or artifacts can resolve outside the repository when run from its root or the default Colab directory.

**Suggested shared addition:** "The install cell installs packages only. Clone or upload the repository into the Colab runtime, including its datasets, and set the kernel's working directory to the repository's `notebooks` directory before continuing. This also keeps relative data, credential, and artifact paths consistent with local runs. Preserve any generated artifacts you need before the runtime is discarded."

**Suggested local note:** "Run with the kernel working directory set to the repository's `notebooks` directory. Opening the notebook in an editor does not by itself guarantee that working directory."

Keep the one-line package-install cells unchanged. A future repository-root discovery helper would be a separate code improvement, not part of this approval.

### NB-08: GPU Guidance Omits Precision Constraints and Overstates Capacity

**Priority:** High. **Decision:** Approved. **Scope:** Markdown with separately identified precision-code issues.

**Location:** [notebooks/sft_full_run.ipynb](notebooks/sft_full_run.ipynb), Cells 1, 3, and 12; [notebooks/sft_lora_run.ipynb](notebooks/sft_lora_run.ipynb), Cells 1 and 3; [notebooks/sft_qlora_run.ipynb](notebooks/sft_qlora_run.ipynb), Cells 1 and 3; [notebooks/grpo_lora_run.ipynb](notebooks/grpo_lora_run.ipynb), Cells 1 and 3. Compare the basic SFT notebook's explicit T4 guidance with these different training paths.

**Finding:** Generic "select a GPU runtime" guidance obscures that GRPO explicitly loads BF16 weights and sets `bf16=True`, while full SFT also selects BF16 on CUDA. A Colab T4 does not provide native BF16 support. The 16 GB claims are not demonstrated memory limits for every model, optimizer, precision mode, or evaluation stage. GRPO's approximately four-hour estimate gives no GPU model or measured run record.

**Suggested hardware prose:** "These settings are a starting point, not a guaranteed fit in 16 GB VRAM. Memory and runtime depend on the model, precision, optimizer, batch size, sequence length, and evaluation workload. Reduce the batch size or use a smaller model when needed. Validate a short run on your selected GPU before committing to a full training run."

**Suggested full SFT/GRPO prep addition:** "The current code uses BF16 on the GPU. Select a BF16-capable runtime, such as a supported Ampere-or-newer Nvidia GPU, or explicitly adapt and test the precision settings before using a T4."

**Suggested GRPO time row:** "Runtime is hardware- and configuration-dependent. This example uses 367 prompts, two epochs, and 12 generations per group without enabling vLLM; benchmark a short run to estimate total time."

**Separate code approval needed:** Full SFT's Cell 14 assigns `torch.dtype` objects to the `bf16` and `fp16` flags rather than Boolean values. Markdown should not imply that selecting a compatible GPU resolves that configuration issue. Precision adaptation and execution testing are outside this Markdown review.

### NB-09: Data Generation Lacks Cost and Retry Warnings

**Priority:** High. **Decision:** Approved. **Scope:** Markdown; bounded retries and request budgets require separate code approval.

**Location:** [notebooks/data_generation.ipynb](notebooks/data_generation.ipynb), Cells 3, 8, and 14; evidence in Cells 9 and 15.

**Finding:** The API workflow defaults to 750 accepted records in batches of up to 10. It continues after request errors and rejected batches without an attempt limit, backoff, or budget limit. The Markdown does not warn about repeated paid calls or persistent failures. The configured model identifier is `gpt-5.6-luna`; this review did not verify its availability or access permissions.

**Suggested addition before generation:** "OpenAI API requests may incur charges. Confirm that `MODEL` is available to your account and supports the Responses API and the requested Structured Outputs schema. Start with a small target and inspect one successful batch before scaling up. The current loop continues until the accepted-record target is met and has no retry or spending cap. Interrupt it if errors persist or no records are being accepted."

**Separate code approval needed:** Add a maximum attempt count, retry/backoff policy, and an explicit stop condition for repeated failures or zero accepted records. No model-name replacement is proposed without checking an available supported model.

### NB-10: GRPO's One-Line QLoRA Claim Does Not Fit Its Model-Loading Path

**Priority:** Medium. **Decision:** Approved. **Scope:** Markdown only; implementing QLoRA is a separate exercise.

**Location:** [notebooks/grpo_lora_run.ipynb](notebooks/grpo_lora_run.ipynb), Cell 23, "Train"; evidence in Cell 24 and the locally installed TRL 1.10.0 trainer source.

**Finding:** The Markdown says `GRPOTrainer` accepts `quantization_config` and QLoRA is a one-line change. The parameter does exist in the inspected version, but the trainer warns and ignores it when passed an already-instantiated model. This notebook constructs its model before creating the trainer. The explanation also conflates two reasons for not keeping a separate reference model: `beta=0` disables the KL reference term; PEFT can recover a reference through adapter handling when a reference is needed.

**Suggested replacement:** "This notebook uses LoRA on a non-quantized base model. To adapt it to QLoRA, configure 4-bit quantization when loading the base model, or use a trainer-supported model-identifier loading path, then verify backend support and training compatibility. Adding `quantization_config` to the trainer does not quantize the model instance already created above. With `beta=0`, this run does not use a KL reference term."

### NB-03: DPO Describes a PiSSA Reference Adapter That Is Not Configured

**Priority:** Medium. **Decision:** Approved. **Scope:** Markdown only.

**Location:** [notebooks/dpo_qlora_run.ipynb](notebooks/dpo_qlora_run.ipynb), Cells 11 and 13, "Prepare the Quantization and LoRA Configs" and "Configure Direct Preference Optimization"; evidence in Cells 12 and 14.

**Finding:** The narrative says TRL preserves an initial PiSSA adapter as a frozen `ref` adapter. The code explicitly uses ordinary LoRA initialization, `init_lora_weights=True`, with `ref_model=None` and `precompute_ref_log_probs=True`. Cell 11 also claims no significant performance or evaluation penalty without presenting a comparison and overgeneralizes reference-model requirements for alternative initializations.

**Suggested replacement for Cell 11:** "Configure a 4-bit base model and LoRA adapters. This example uses standard LoRA initialization: the B matrix starts at zero, so the initial adapter contributes no weight update. Alternative initializations such as PiSSA and LoftQ require different setup and are not demonstrated here."

**Suggested replacement for Cell 13:** "Configure DPO with `beta=0.1` and precomputed reference log probabilities. With the PEFT reference path, the unchanged base model provides the reference behavior; precomputation avoids repeating reference scoring on every training step."

### NB-04: Adapter SFT Markdown Still Describes Full Fine-Tuning

**Priority:** Medium. **Decision:** Approved. **Scope:** Markdown only.

**Location:** [notebooks/sft_qlora_run.ipynb](notebooks/sft_qlora_run.ipynb), Cells 1, 12, and 14; [notebooks/sft_lora_run.ipynb](notebooks/sft_lora_run.ipynb), Cells 12 and 14.

**Finding:** The QLoRA title says only LoRA, and its introduction describes a full fine-tuning path, a 1B-3B target, and LoRA/QLoRA being deferred to a later notebook. It actually selects Qwen3-8B and configures 4-bit adapter training. Both adapter notebooks describe `SFTConfig` as a full fine-tuning run. The prose also uses `LoRAConfig` instead of the actual API name, `LoraConfig`.

**Suggested QLoRA title and introduction:** "# Taco Alley Complaint Structuring with QLoRA SFT" followed by: "Train LoRA adapters on a 4-bit Qwen3-8B base model to convert incoming complaints into structured JSON with `category`, `sub_category`, and `tone_urgency`. Quantization reduces base-model memory use; the adapters are the trainable parameters."

**Suggested configuration descriptions:** For LoRA: "Build `LoraConfig` for parameter-efficient SFT on Qwen3-1.7B." For QLoRA: "Build `BitsAndBytesConfig` for 4-bit loading and `LoraConfig` for adapter training." Replace "full fine-tuning run" with "LoRA adapter training run" or "QLoRA adapter training run", respectively. Hardware claims are addressed separately below.

### NB-05: Dataset Exploration Overstates Its Local-File Setup

**Priority:** Medium. **Decision:** Approved. **Scope:** Markdown only.

**Location:** [notebooks/datasets.ipynb](notebooks/datasets.ipynb), Cells 8 and 17, "Review dataset features and sample rows" and "Load a dataset from a local file"; evidence in Cell 18.

**Finding:** Cell 8 says to inspect "both datasets" before the local dataset is loaded. Cell 17 says the notebook creates a file and is self-contained. The code first reuses the supplied sentiment file, only writes fallback examples if it is missing, and does not create the parent directory. It also assumes a working directory immediately beneath the repository root.

**Suggested Cell 8 edit:** "Inspect the online dataset's schema, column names, and sample records. After the local-file section, repeat these checks on `file_dataset`."

**Suggested Cell 17 replacement:** "Load the supplied `datasets/sample_sentiment.jsonl` file. If it is absent, the next cell writes a small fallback dataset; the parent `datasets/` directory must already exist. To use a different local CSV or JSONL file, update the path and choose the corresponding dataset loader."

### NB-06: GRPO Probe Commentary Misstates the Reward Values

**Priority:** Medium. **Decision:** Approved. **Scope:** Markdown only.

**Location:** [notebooks/grpo_lora_run.ipynb](notebooks/grpo_lora_run.ipynb), Cell 18, commentary after the reward probes; evidence in Cells 13, 15, and 17.

**Finding:** The wrong-literal probe runs `COUNT(*)`; a no-match count returns one row containing zero, not an empty result set. A formatted, executable but incorrect query still earns the format and execution rewards: `0.15 + 0.25 = 0.40` before any correctness or overlap reward. The destructive probe receives a weighted penalty of `-0.15`, so "strongly negative" is imprecise. The paragraph also attributes dataset filtering to a generator script that is not included in the repository.

**Suggested replacement:** "Inspect each reward component, not just the total. The wrong-literal query can execute successfully and earn formatting and execution credit while receiving no correctness credit; `COUNT(*)` returns a zero count rather than an empty result set. The destructive probe is rejected before execution and receives a weighted total of -0.15. Gold answers that are zero or empty need special scrutiny because unrelated incorrect queries may return the same result. Audit these cases in the supplied dataset; this repository does not include the dataset-generation script."

### NB-11: DPO Reward Accuracy Is Easy to Mistake for Response Quality

**Priority:** Medium. **Decision:** Approved. **Scope:** Markdown only.

**Location:** [notebooks/dpo_qlora_run.ipynb](notebooks/dpo_qlora_run.ipynb), Cells 15, 17, and 19.

**Finding:** Cell 15 correctly introduces a reference-relative metric, but Cell 17 drops that qualification when interpreting values above 0.5. Reward accuracy is not the fraction of helpful generated responses or necessarily the raw fraction where the chosen completion has a higher policy likelihood. The initial policy and reference can tie; the inspected TRL implementation uses a strict greater-than comparison, so ties count as zero rather than chance-level accuracy.

**Suggested replacement for Cell 15's explanation:** "DPO reward accuracy measures how often the chosen response has a higher reference-relative reward than the rejected response. At initialization, the policy and reference may tie, yielding zero accuracy under the strict comparison. The first evaluation also precomputes reference log probabilities for the evaluation split."

**Suggested training/evaluation addition:** "A value above 0.5 means positive reference-relative reward margins for a majority of evaluated preference pairs. It is not an absolute response-quality score. Compare held-out metrics and inspect generated responses before concluding that customer-service quality improved."

### NB-12: GRPO Advantage and Diagnostic Explanations Are Too Absolute

**Priority:** Medium. **Decision:** Approved. **Scope:** Markdown only.

**Location:** [notebooks/grpo_lora_run.ipynb](notebooks/grpo_lora_run.ipynb), Cells 1, 12, 14, 23, and 26.

**Finding:** Advantage is described only as reward minus group mean. The inspected TRL version defaults to group reward scaling, adding division by group standard deviation plus an epsilon. Several explanations turn useful signals into definitive diagnoses: zero reward variance does not uniquely identify a deterministic policy, low execution reward does not uniquely identify a prompt error, and an entropy drop alone does not establish collapse. Fixed rules such as "16 is better" and "keep temperature at 1.0" lack a demonstrated comparison. The batch divisibility rule omits device count and generation-batch overrides.

**Suggested advantage wording:** "GRPO centers each completion's reward against its group's mean. With the default group scaling, it also divides by the group's reward standard deviation plus a small epsilon. Equal-reward groups provide no reward-based policy-gradient signal; check whether this reflects consistent success, consistent failure, or insufficient reward discrimination."

**Suggested configuration wording:** "This single-GPU configuration uses 12 generations, batch size 1, and 12 accumulation steps. The effective generation batch must be compatible with the group size; include all devices and any generation-batch overrides when changing it. Group size and sampling temperature trade generation cost against diversity, so compare short runs rather than treating the current values as universal optima."

**Suggested metric-table edits:** Replace "means" or "is" diagnoses with "can indicate" and list checks. For flat result-match reward, inspect generated SQL, component rewards, data, prompt, and optimization settings. For high zero-standard-deviation fraction, inspect actual rewards to distinguish all-correct from all-wrong groups. For entropy drops, check held-out accuracy and examples for repetitive or degenerate output before declaring collapse.

### NB-13: GRPO Evaluation and SQL Safety Need Explicit Limits

**Priority:** Medium. **Decision:** Rejected. **Scope:** Markdown only; stronger isolation or uncapped result comparison would require separate code approval.

**Location:** [notebooks/grpo_lora_run.ipynb](notebooks/grpo_lora_run.ipynb), Cells 1, 6, 10, and 12; implementation in Cells 7, 13, and 20.

**Finding:** The notebook reasonably compares query results instead of SQL strings, but the prose omits that execution fetches at most 200 rows and floating-point values are rounded to two decimals. It presents held-out-template improvement as evidence that the model learned to read the schema, which is stronger than this evaluation alone establishes. Read-only connections, query-only mode, a text filter, and time/step limits are useful demo safeguards, not a complete isolation boundary for arbitrary model-generated SQL.

**Suggested result-comparison addition:** "Execution accuracy here is equality of normalized query outputs on this database snapshot, limited to the first 200 fetched rows. Floats are rounded to two decimals, and ordered comparisons are selected using an `ORDER BY` check in the gold SQL. Matching results on this snapshot does not prove that two queries are semantically equivalent on every database."

**Suggested split explanation:** "The parameter-held-out split tests new values within familiar question templates. The template-held-out split tests different question shapes in this dataset and is a stronger generalization check. Report both; neither alone proves general SQL competence or transfer to unseen schemas."

**Suggested safety addition:** "Use only the supplied disposable demo database in an isolated environment. These safeguards limit writes and query execution but are not a production SQL sandbox; read-only access does not protect confidential data from being read."

### NB-14: Synthetic Data Quality and Output Lineage Are Underexplained

**Priority:** Medium. **Decision:** Rejected. **Scope:** Markdown only.

**Location:** [notebooks/data_generation.ipynb](notebooks/data_generation.ipynb), Cells 1, 7, 12, 14, and 16; evidence in Cells 11, 13, 15, and 17. Related input: [notebooks/dpo_qlora_run.ipynb](notebooks/dpo_qlora_run.ipynb), Cell 6.

**Finding:** "Clean records" can imply semantic quality checks that are not implemented. Validation checks schema constraints and different chosen/rejected strings; deduplication compares normalized fingerprints, not semantic similarity. Seed records are carried into the final dataset without being passed through the generated-record validation loop. The output file is overwritten, and its name differs from the curated preference file DPO consumes, with no documented handoff.

**Suggested validation wording:** "Validate generated records against the schema, reject identical chosen/rejected strings, and remove duplicates using a case- and whitespace-normalized fingerprint. These checks do not establish factual accuracy, semantic uniqueness, or that the chosen response is genuinely better. Review a sample manually before training. Seed records are assumed to have been curated separately."

**Suggested save-section addition:** "This cell overwrites `datasets/synthetic_data.csv` with the combined seed and accepted generated records. Change `OUTPUT_PATH` to preserve an existing file. The DPO notebook reads the separate supplied `datasets/synthetic_data_dpo.csv`; this notebook does not automatically update that training input."

### NB-15: Basic SFT Evaluation and Upload Side Effects Need Clearer Labels

**Priority:** Medium. **Decision:** Approved. **Scope:** Markdown only.

**Location:** [notebooks/basic_sft_example.ipynb](notebooks/basic_sft_example.ipynb), Cells 13, 17, 20, and 23; evidence in Cells 14, 19, 22, and 24.

**Finding:** The evaluation prose omits the answer-extraction heuristic and may be read as a general GSM8K benchmark result. The code evaluates the same first 100 test examples before and after training, using greedy decoding and the last number in the generated answer. "This final section" is also inaccurate because upload and cleanup follow. Uploads are real remote writes, and the repository visibility is not explicitly configured. The cleanup cell deletes one model reference and clears caches but does not guarantee that all kernel-held GPU resources are released.

**Suggested evaluation text:** "Evaluate the same first 100 GSM8K test questions before and after training using greedy decoding. The scorer compares the last number extracted from the generated answer with the reference final answer. This is a small diagnostic sample with a heuristic scorer, not a full GSM8K benchmark result. Inspect extraction failures and sample answers alongside the accuracy."

**Suggested upload heading and text:** "## Upload the trained model (optional)" followed by: "The following cells publish model, tokenizer, and model-card files to your Hugging Face account under `RUN_NAME`. Verify the destination and repository visibility before running them, and ensure the uploaded files contain no credentials or sensitive data. Skip this section to keep the model local."

**Suggested cleanup text:** "Release the model reference used here and clear unused CUDA cache blocks. Other live references can keep memory allocated; restart the kernel when a full reset is needed."

### NB-16: Complaint SFT Notebooks Need Navigation for Evaluation and Export

**Priority:** Low. **Decision:** Approved. **Scope:** Markdown only; add headings without moving or changing code cells.

**Location:** [notebooks/sft_full_run.ipynb](notebooks/sft_full_run.ipynb), Cell 15 and the code in Cells 16-19; [notebooks/sft_lora_run.ipynb](notebooks/sft_lora_run.ipynb), Cells 12 and 17 and code in Cells 18-22; [notebooks/sft_qlora_run.ipynb](notebooks/sft_qlora_run.ipynb), Cells 12 and 19 and code in Cells 20-25.

**Finding:** Long code sequences combine setup, baseline evaluation, training, saving, reload, and post-training evaluation under very broad descriptions. The LoRA notebook does not explain its PiSSA initialization and conversion on save. The QLoRA narrative does not orient the learner to its separate LoftQ initialization step. Readers also need to distinguish a full-model checkpoint from adapter-only output. The adapter notebooks do not perform the full SFT notebook's explicit pre-training baseline comparison.

**Suggested headings at the corresponding boundaries:** "Evaluate the baseline" where an actual baseline call exists; "Train and inspect loss"; "Save the model" for full SFT or "Save the adapter" for LoRA/QLoRA; and "Reload and evaluate". Where one existing cell contains multiple stages, describe them in its preceding Markdown rather than splitting code without approval.

**Suggested LoRA configuration addition:** "This example uses PiSSA initialization. The initial PiSSA weights are saved for conversion when exporting the trained adapter as a standard LoRA update."

**Suggested QLoRA configuration addition:** "After constructing the quantized model and attaching LoRA adapters, the notebook applies a LoftQ weight replacement before training. Distinguish this initialization step from the optimizer's training updates."

**Suggested output notes:** Full SFT saves a complete model beneath `artifacts/sft_full_run/final_model`. LoRA saves the converted adapter beneath `artifacts/sft_lora_run/adapters`; QLoRA saves an adapter beneath `artifacts/sft_qlora_run/adapters_qlora`. Adapter outputs still require the matching base model. State that the adapter notebooks report post-training metrics, not an in-notebook before/after baseline delta.

### NB-17: Headings, Terminology, and Navigation Vary Across the Series

**Priority:** Low. **Decision:** Approved. **Scope:** Markdown only.

**Location:** All eight notebooks, particularly the SFT "Demonstration:" headings and blockquotes, GRPO's numbered headings and empty Cell 25, and the basic SFT introduction's unlinked README reference.

**Finding:** The series mixes sentence case and title case, numbered and unnumbered sections, ordinary explanation and blockquote-styled explanation, and broad versus very terse step descriptions. GRPO numbering starts at "2" after the shared prep heading. The full SFT and adapter introductions jump from H1 to H3 for hardware guidance. "LoRA based", "LoRA-based", `LoRAConfig`, and `LoraConfig` are inconsistent; the real class name is `LoraConfig`. Acronyms are not consistently expanded on first use.

**Suggested convention:** Keep one H1 title, H2 major workflow sections in sentence case, and H3 subsections only beneath H2. Preserve "Google Colab prep" and the simple install cells. Use ordinary paragraphs for explanations; reserve blockquotes for warnings or quotations. Expand supervised fine-tuning (SFT), low-rank adaptation (LoRA), quantized LoRA (QLoRA), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO) on first relevant use. Use code formatting for actual API identifiers and "16 GB" for memory units.

**Suggested targeted edits:** Remove the numeric prefixes from GRPO sections instead of maintaining a second ordering system; remove its empty Markdown Cell 25; link to the root [README.md](README.md) using `../README.md` as the target inside notebooks; add links to explicitly named predecessor or successor notebooks instead of "the later notebook". Add the local-skip sentence to the already-simple basic SFT and dataset prep descriptions.

**Approval choice:** Keep "Demonstration:" if those headings intentionally mirror course chapters. Removing that prefix is optional editorial cleanup, not an accuracy fix. Preserve the scenario and instructional detail rather than rewriting every notebook into identical prose.

### NB-18: Optional Model Download Guidance Is Too Categorical

**Priority:** Low. **Decision:** Approved. **Scope:** Markdown only.

**Location:** [notebooks/sft_qlora_run.ipynb](notebooks/sft_qlora_run.ipynb), Cell 14.

**Finding:** "The progress bars don't work very well" is a broad environment-dependent assertion. The bare CLI command also does not say which Python environment should contain the `hf` executable or why pre-downloading is optional.

**Suggested replacement:** "Optional: pre-download Qwen3-8B to separate download time from model loading. Use `uv run hf download \"Qwen/Qwen3-8B\"` from the local project environment, or `%pip`-prepared Colab's `!hf download \"Qwen/Qwen3-8B\"` in a notebook code cell. Skip this step if the model is already cached; model loading can also download it automatically. Allow enough disk space for the original model files even though training uses 4-bit quantization."

Keep any executable notebook cell addition separate from approval of this Markdown replacement.

## Coverage

| Notebook | Relevant findings | Content that remains useful |
| --- | --- | --- |
| [notebooks/basic_sft_example.ipynb](notebooks/basic_sft_example.ipynb) | NB-02, NB-07, NB-15, NB-17 | Model/dataset identification, pre/post sample evaluation flow, simple install pattern |
| [notebooks/data_generation.ipynb](notebooks/data_generation.ipynb) | NB-02, NB-07, NB-09, NB-14, NB-17 | Workflow outline and Responses API Structured Outputs description |
| [notebooks/datasets.ipynb](notebooks/datasets.ipynb) | NB-05, NB-07, NB-17 | Online loading, schema exploration, shuffle/map, and subset examples |
| [notebooks/dpo_qlora_run.ipynb](notebooks/dpo_qlora_run.ipynb) | NB-03, NB-07, NB-11, NB-17 | Preference-pair structure, separate completions, and adapter/base-model distinction |
| [notebooks/grpo_lora_run.ipynb](notebooks/grpo_lora_run.ipynb) | NB-01, NB-07, NB-08, NB-10, NB-06, NB-12, NB-13, NB-17 | Taco Alley scenario, reward components, probe exercise, and before/after evaluation |
| [notebooks/sft_full_run.ipynb](notebooks/sft_full_run.ipynb) | NB-07, NB-08, NB-16, NB-17 | Complaint schema and baseline/full-training comparison |
| [notebooks/sft_lora_run.ipynb](notebooks/sft_lora_run.ipynb) | NB-04, NB-07, NB-08, NB-16, NB-17 | Complaint task and adapter training progression |
| [notebooks/sft_qlora_run.ipynb](notebooks/sft_qlora_run.ipynb) | NB-04, NB-07, NB-08, NB-16, NB-17, NB-18 | Complaint task and quantized adapter training progression |

All eight Colab badges target their matching notebook in this repository; no badge-target edit is proposed. The GRPO training split contains 367 records as stated; the parameter and template evaluation files contain 57 and 41 records, respectively.

## Verification and Limits

This is a static review of all Markdown cells in the eight notebooks, nearby implementation, and supplied files. Trainer-specific statements were checked against the locally installed TRL 1.10.0 source, not against a running Colab environment. Colab's unpinned installations can resolve another version, so those details need rechecking when implementing or testing changes.

No package installations, API requests, uploads, or GPU training were run. Runtime, memory, model availability, external link availability, and behavior on alternative package versions remain unverified. Notebook JSON was parsed to inspect cells; cell numbers and approval scope must be rechecked before applying edits after notebook changes.

No notebook changes are authorized by this document's creation. Approve Markdown findings individually, and explicitly approve any associated code work separately.
