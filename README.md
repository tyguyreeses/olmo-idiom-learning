# OLMo Idiom Comprehension Study

## 1. Project Overview

Build a complete, reproducible research pipeline to investigate how idiomatic expression comprehension develops throughout the pretraining of a large language model.

The project will use intermediate training checkpoints of **OLMo-2-7B Base**, published by the Allen Institute for AI (Ai2), to evaluate how accurately the model understands and explains idiomatic expressions at different points in its training.

The experiment will use idiom datasets containing idiomatic expressions and reference explanations across multiple languages. The model will receive an idiom and be instructed to explain its intended meaning in the corresponding language.

The generated explanations will then be evaluated against the reference explanations using a semantic accuracy rubric.

The primary objective is to determine how idiomatic comprehension changes as the model progresses through training.

### Primary research question

At what point in its training does an LLM learn idiomatic expressions to a satisfactory level?

### Secondary research questions

1. How does idiomatic comprehension improve throughout pretraining?
2. Does idiomatic comprehension develop at different rates across languages?
3. Does idiomatic comprehension improve steadily, or are there identifiable points of significant improvement?
4. How does model performance vary depending on the idiom and language?

### Experimental scope

This is a university course research project, not a large-scale research benchmark. Prioritize:

- A functional and reproducible experiment.
- Reliable data collection.
- Consistent prompting.
- Meaningful semantic evaluation.
- Clear quantitative results.
- Efficient execution on a university GPU cluster.

Avoid unnecessary architectural complexity or infrastructure.

---

# 2. Technology Stack

Use the following technologies:

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Deep learning | PyTorch |
| Model loading | Hugging Face Transformers |
| Model repository access | Hugging Face Hub |
| Dataset processing | Pandas |
| Data storage | JSONL and CSV |
| Configuration | YAML |
| Experiment orchestration | Slurm |
| Statistical analysis | Pandas, NumPy, SciPy |
| Visualization | Matplotlib |
| Testing | Pytest |
| Evaluation | LLM-as-a-judge, optional BERTScore |

Use standard, well-supported libraries wherever possible.

Do not introduce a database, web server, frontend, distributed computing framework, or other unnecessary infrastructure.

---

# 3. Model and Checkpoint Selection

## Primary model

Use:

`allenai/OLMo-2-1124-7B`

Hugging Face repository:

https://huggingface.co/allenai/OLMo-2-1124-7B

This is the base pretrained model, not the instruction-tuned version.

Do not use:
- OLMo-2-1124-7B-Instruct
- OLMo-2-1124-7B-SFT
- OLMo-3 models
- Other model families

The purpose is to investigate the development of idiomatic knowledge during pretraining rather than instruction-following performance.

## Intermediate checkpoints

The experiment should evaluate eight checkpoints from the same OLMo-2-7B training run.

The checkpoints should represent different points throughout pretraining.

Do not assume checkpoint revision names or training token counts.

Implement a checkpoint discovery mechanism using Hugging Face Hub where possible. Allow the user to explicitly define the eight checkpoint revisions in a configuration file.

The checkpoint manifest should contain:

```yaml
model:
  repository: allenai/OLMo-2-1124-7B

checkpoints:
  - id: checkpoint_0
    revision: "<verified_revision>"
    training_step: null
    training_tokens: null

  - id: checkpoint_1
    revision: "<verified_revision>"
    training_step: null
    training_tokens: null

  # Six additional checkpoints
```

The user will populate the verified revision names and metadata after inspecting the repository.

### Checkpoint selection requirements

1. Select eight checkpoints from the same training run.
2. Distribute them throughout the available training progression.
3. Include the final pretrained checkpoint.
4. Record training step and token count whenever available.
5. Record training-stage transitions.
6. Do not mix checkpoints from different training runs.
7. Do not silently substitute a checkpoint if a revision is unavailable.

The final checkpoint should serve as the final performance baseline.

The primary independent variable is training progress, preferably measured in tokens processed rather than only training steps.

---

# 4. Dataset Management

The experiment will use two datasets:

- MIDAS
- PETCI

The user already has access to these datasets.

Do not download alternative datasets or assume that the datasets are publicly accessible.

Do not modify the original dataset files.

## Dataset requirements

Each dataset should be converted into a standardized internal representation.

The pipeline should support:

- Multiple languages.
- Idiomatic expressions.
- Reference explanations.
- Optional dataset identifiers.
- Optional metadata such as idiom category or literal meaning.

The supplied schemas are documented below. The source JSON files must not be modified; use the repository-root script `normalize_datasets.py` to create processed JSONL files.

Do not fabricate missing reference explanations.

### Internal dataset representation

Normalize records into the following structure:

```json
{
  "idiom_id": "unique_id",
  "dataset": "PETCI",
  "language": "zh",
  "target_language": "en",
  "idiom": "一波未平，一波又起",
  "reference_explanation": "suffer a string of reverses",
  "metadata": {
    "reference_explanation_language": "en"
  }
}
```

Required fields:

- `idiom_id`
- `dataset`
- `language`
- `idiom`
- `reference_explanation`
- `target_language` (the model's output language; equals `language` for MIDAS and is `en` for PETCI)

Optional source fields can be retained in `metadata`, including `reference_explanation_language`.

### Dataset processing

Implement:

1. Dataset loading.
2. Schema validation.
3. Language identification or mapping using configured dataset metadata.
4. Duplicate detection.
5. Missing-value detection.
6. Normalization into the internal representation.
7. Dataset statistics.

Produce a dataset summary containing:

- Total number of idioms.
- Number of idioms per dataset.
- Number of records per idiom language and target-language pair.
- Number of missing explanations.
- Number of duplicate records.

Do not silently discard problematic records. Log exclusions and their reasons.

### Dataset normalization

Run the normalizer from the repository root:

```bash
python normalize_datasets.py
```

Outputs are written to `data/processed/`: `midas.jsonl`, `petci.jsonl`, `rejected.jsonl`, and `summary.json`. It trims strings and applies Unicode NFC normalization without changing idiom casing or punctuation. Missing required fields are recorded with source file, record index, ID, and exclusion reason. Exact duplicate idiom-and-reference pairs are retained and annotated with `metadata.duplicate_of`, rather than automatically merged because repeated idioms can have different senses.

Each normalized record contains `idiom_id`, `dataset`, `language`, `target_language`, `idiom`, `reference_explanation`, and `metadata`. `language` is the idiom language; `target_language` is the language requested for model generation. MIDAS aliases and context are retained in metadata. PETCI's `gold` field is the primary reference; its `human` and `machine` values are preserved as alternate references. PETCI uses `language: "zh"`, `target_language: "en"`, and `reference_explanation_language: "en"`. Compare and report PETCI separately from MIDAS same-language tasks.

---

# 5. Prompting and Generation

## Objective

For every idiom in the normalized dataset, generate one explanation for each checkpoint.

The model should explain the idiom's intended meaning in the record's `target_language`. For MIDAS this is the idiom's original language. For PETCI, the idiom is Chinese and the explanation target is English; treat and report PETCI as a separate Chinese-idiom-to-English task.

The prompt itself should be written in English.

The prompt should explicitly specify the desired output language, using `target_language` rather than assuming it matches the idiom language.

For example, an English prompt may instruct the model to explain an idiom in Mandarin Chinese, or to explain a Chinese idiom in English for PETCI.

## Prompt consistency

Use the exact same prompt template across all checkpoints.

The following must remain constant:

- Prompt wording.
- Few-shot examples.
- Idiom input.
- Target language.
- Generation parameters.
- Output extraction rules.

This is important because the experiment should measure differences in model performance, not differences in prompting.

## Few-shot prompting

Support zero-shot and one-shot/two-shot prompting.

The preferred initial configuration is one-shot prompting, with the possibility of two-shot prompting.

The examples must:

- Be written in the appropriate target language.
- Demonstrate the desired explanation format.
- Avoid unnecessary additional commentary.
- Be identical across checkpoints.
- Not appear in the evaluation dataset.

Allow examples to be configured separately by language.

Do not automatically select examples from the evaluation dataset.

## Prompt template

Use a configurable prompt template similar to:

```text
You are explaining the meaning of idiomatic expressions.

Explain the intended meaning of the following idiom in {target_language}.

Describe its figurative meaning rather than interpreting it literally.

Return only the explanation. Do not include the idiom,
a translation, introductory text, or additional commentary.

Example:
Idiom: {example_idiom}
Explanation: {example_explanation}

Now explain:

Idiom: {idiom}
Explanation:
```

The actual prompt and examples should be stored in configuration files rather than hardcoded throughout the codebase.

## Important base-model consideration

OLMo-2-7B Base is a pretrained causal language model, not an instruction-tuned chat model.

Do not assume it supports a chat template.

Use standard causal language model prompting and completion.

The model should receive the prompt as text and generate a continuation.

Implement output extraction that removes the original prompt and retains the generated explanation.

Do not use an instruction-tuning or chat template unless explicitly verified to be supported and justified.

## Generation parameters

Make generation parameters configurable.

Suggested initial settings:

```yaml
generation:
  max_new_tokens: 128
  temperature: 0.0
  do_sample: false
  batch_size: 4
  precision: bfloat16
```

These are starting values, not immutable requirements.

Use deterministic generation where possible.

Use the same generation settings for every checkpoint.

Do not allow one checkpoint to use different sampling settings from another.

Support batching to improve GPU throughput.

Handle:

- Empty outputs.
- Generation errors.
- CUDA out-of-memory errors.
- Tokenization errors.
- Unexpectedly long outputs.
- Interrupted jobs.

Log errors without silently dropping idioms.

---

# 6. Generation Output Format

Save all generated explanations in JSONL format.

Use one record per idiom per checkpoint.

Example:

```json
{
  "experiment_id": "olmo2_idiom_study_v1",
  "checkpoint_id": "checkpoint_0",
  "model_repository": "allenai/OLMo-2-1124-7B",
  "model_revision": "verified_revision",
  "training_step": 1000,
  "training_tokens": 5000000000,
  "dataset": "MIDAS",
  "idiom_id": "en_001",
  "language": "en",
  "idiom": "kick the bucket",
  "reference_explanation": "To die.",
  "prompt": "Full prompt...",
  "generated_explanation": "To pass away or die.",
  "generation_parameters": {
    "temperature": 0.0,
    "max_new_tokens": 128
  },
  "generation_status": "success"
}
```

Values above are illustrative.

### Output requirements

- Preserve the original reference explanation.
- Preserve the full generated explanation.
- Record the exact checkpoint revision.
- Record training metadata.
- Record the language.
- Record the prompt or enough information to reconstruct it exactly.
- Record generation parameters.
- Record errors and unsuccessful generations.

Do not overwrite existing results by default.

Each checkpoint should have its own output file:

```text
results/
  generations/
    checkpoint_0.jsonl
    checkpoint_1.jsonl
    checkpoint_2.jsonl
    checkpoint_3.jsonl
    checkpoint_4.jsonl
    checkpoint_5.jsonl
    checkpoint_6.jsonl
    checkpoint_7.jsonl
```

Implement resume functionality so that already-completed idioms are not unnecessarily regenerated after an interrupted job.

Use a stable record identifier based on dataset, idiom ID, and checkpoint ID.

---

# 7. Evaluation Methodology

Generation and evaluation must be separate processes.

The generation jobs should not perform expensive evaluation.

Once all checkpoint outputs have been generated, run the evaluation pipeline independently.

## Primary evaluation: LLM-as-a-judge

Use an LLM to compare each generated explanation against the corresponding reference explanation.

The evaluator should assess semantic accuracy rather than exact wording.

The evaluator should determine whether the generated explanation communicates the intended meaning of the idiom.

The evaluation model should ideally be independent of the OLMo model being evaluated.

If using an OLMo model as the evaluator, use the same evaluator checkpoint for all generated explanations and acknowledge the possibility of evaluator bias.

Do not use different evaluator models for different checkpoints.

## Scoring rubric

Use a 0–4 integer scale.

| Score | Description |
|---|---|
| 0 | Completely incorrect, unrelated, or fundamentally misunderstands the idiom |
| 1 | Mostly incorrect; captures little of the intended meaning |
| 2 | Partially correct; captures some of the intended meaning but misses important information |
| 3 | Mostly correct; communicates the intended meaning with minor omissions or inaccuracies |
| 4 | Fully correct; accurately communicates the intended meaning |

The evaluator should follow these principles:

1. Evaluate semantic meaning, not lexical similarity.
2. Accept alternative explanations that communicate the same meaning.
3. Do not penalize different wording.
4. Do not penalize reasonable differences in explanation length.
5. Penalize literal interpretations when they fail to communicate the figurative meaning.
6. Penalize invented or incorrect details.
7. Evaluate in the context of the idiom's language.
8. Do not reward explanations simply because they are longer.

## Evaluation prompt

Create a configurable evaluation prompt similar to:

```text
You are evaluating the semantic accuracy of an explanation
of an idiomatic expression.

You will receive:
1. An idiomatic expression.
2. The language of the expression.
3. A reference explanation.
4. A generated explanation.

Determine how accurately the generated explanation communicates
the intended meaning described by the reference.

Evaluate semantic accuracy, not wording similarity.

Alternative wording is acceptable if it communicates the same meaning.

Do not penalize stylistic differences or reasonable differences
in explanation length.

Score the explanation using the following rubric:

0: Completely incorrect or unrelated.
1: Mostly incorrect.
2: Partially correct.
3: Mostly correct, with minor omissions or inaccuracies.
4: Fully accurate.

Idiom: {idiom}
Language: {language}
Reference explanation: {reference}
Generated explanation: {generated}

Return only valid JSON:

{
  "score": 0,
  "reasoning": "Brief explanation of the score."
}
```

The evaluator must return a valid integer score between 0 and 4.

Validate evaluator outputs before saving.

If the response is malformed, retry a limited number of times and log the failure.

Do not silently convert malformed responses into scores.

## Evaluator configuration

Support an external LLM API or a locally hosted evaluator.

Keep provider-specific implementation separate from the evaluation pipeline.

Do not hardcode API keys.

Use environment variables for credentials.

Allow the evaluator model to be configured independently of the generation model.

## Human validation

Support optional human evaluation.

The project should make it easy to export a random sample of generated explanations for manual scoring.

Ideally, manually score a subset of explanations across different checkpoints and languages.

Compare human scores against LLM evaluator scores using an appropriate agreement or correlation measure.

This is particularly important because the evaluator itself may have biases.

## Optional supplementary metrics

Implement optional BERTScore if practical.

BERTScore can provide a supplementary semantic similarity measurement, but it should not replace the primary accuracy rubric.

Do not use BLEU as the primary evaluation metric.

BARTScore may be explored as an additional metric, but it is not required.

The primary reported accuracy measure should remain the LLM judge's 0–4 score.

---

# 8. Evaluation Output Format

Save evaluation results separately from generation results.

One record per generated explanation:

```json
{
  "checkpoint_id": "checkpoint_0",
  "idiom_id": "en_001",
  "dataset": "MIDAS",
  "language": "en",
  "generated_explanation": "To pass away or die.",
  "reference_explanation": "To die.",
  "score": 4,
  "reasoning": "The explanation accurately communicates the figurative meaning.",
  "evaluator_model": "evaluator_name",
  "evaluation_status": "success"
}
```

Save to:

```text
results/
  evaluations/
    checkpoint_0_evaluations.jsonl
    checkpoint_1_evaluations.jsonl
    ...
```

The evaluation records must be joinable with generation records using stable identifiers.

---

# 9. Statistical Analysis

The final objective is to calculate average idiom comprehension scores per checkpoint and language.

Implement an analysis pipeline that reads all successful generation and evaluation records.

## Primary metrics

Calculate:

1. Mean score per checkpoint.
2. Mean score per language.
3. Mean score per checkpoint-language combination.
4. Mean score per dataset.
5. Number of successful and failed generations.
6. Number of successfully evaluated records.

Convert the 0–4 score into a percentage for presentation:

```text
normalized_score = (mean_score / 4) * 100
```

Clearly label this as a normalized rubric score, not a conventional percentage of correct answers.

## Required visualizations

Generate the following plots:

### A. Overall training progression

- X-axis: training tokens processed.
- Y-axis: average idiom comprehension score.
- One data point per checkpoint.

### B. Language-specific progression

- X-axis: training tokens processed.
- Y-axis: average idiom comprehension score.
- One line per language.

### C. Final performance by language

- X-axis: language.
- Y-axis: average score.
- Compare the final checkpoint across languages.

### D. Score distribution

Show the distribution of 0–4 scores across checkpoints.

### E. Dataset comparison

Compare results from MIDAS and PETCI where applicable.

Save plots as PNG files.

## Statistical considerations

Calculate sample sizes alongside averages.

Where practical, calculate confidence intervals using idiom-level bootstrap resampling.

If comparing checkpoints, preserve idiom identity when calculating differences because the same idioms are evaluated at every checkpoint.

Do not treat every model response as an independent observation when the same idioms are repeated across checkpoints.

Avoid making strong statistical claims from small samples.

## Aggregation requirements

Generate a master CSV containing:

```text
checkpoint_id
training_step
training_tokens
language
dataset
sample_count
mean_score
normalized_score
standard_deviation
```

Also generate a summary table containing one row per checkpoint-language combination.

---

# 10. Slurm Integration

The project must support execution on a university GPU cluster using Slurm.

The exact cluster configuration is currently unknown.

Do not assume:

- A specific GPU partition.
- A specific GPU request syntax.
- A specific account or allocation.
- A specific module system.
- A specific filesystem.
- A specific CUDA version.

All cluster-specific values must be configurable.

## Job array design

Use one Slurm array task per checkpoint.

The intended configuration is:

```bash
#SBATCH --array=0-7%4
```

This creates eight independent tasks while allowing at most four to run concurrently.

Each task:

1. Reads its assigned checkpoint from the checkpoint manifest.
2. Loads the model.
3. Loads the normalized dataset.
4. Generates explanations for all assigned idioms.
5. Saves its own output file.
6. Exits cleanly.

The mapping between array task ID and checkpoint must come from the checkpoint manifest, not from hardcoded model revision names.

## Initial resource estimates

Use these as configurable starting values:

| Resource | Initial request |
|---|---|
| Nodes | 1 |
| GPUs | 1 |
| CPUs | 4 |
| System RAM | 32 GB |
| GPU memory | Prefer 24 GB or more |
| Wall time | 4 hours, subject to benchmarking |
| Concurrent tasks | 4 |

These are initial estimates, not guaranteed cluster requirements.

Do not assume that every GPU supports BF16. Detect or configure supported precision.

## Slurm script

Create:

`slurm/generate_array.sh`

The script should:

- Accept the array task ID.
- Load the configured environment.
- Set Hugging Face cache directories.
- Execute the generation script.
- Write separate logs per task.
- Preserve exit codes.
- Handle errors cleanly.

Example structure:

```bash
#!/bin/bash

#SBATCH --job-name=olmo-idioms
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# GPU request and partition must be configured
# according to the university cluster.

python -m src.generate \
    --checkpoint-index "$SLURM_ARRAY_TASK_ID"
```

This is a template. The agent should not assume it can be submitted without cluster-specific adjustments.

## Job monitoring

Document commands for:

- Submitting the array.
- Checking job status.
- Viewing logs.
- Cancelling jobs.
- Resubmitting failed tasks.

Include common Slurm commands such as:

```bash
sbatch slurm/generate_array.sh
squeue -u "$USER"
scancel JOB_ID
```

Do not automatically submit jobs without explicit user action.

---

# 11. Repository Structure

Build the repository with the following organization:

```text
olmo-idiom-study/
│
├── README.md
├── normalize_datasets.py       # normalizes supplied MIDAS and PETCI JSON
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── configs/
│   ├── experiment.yaml
│   ├── datasets.yaml
│   ├── checkpoints.yaml
│   ├── generation.yaml
│   ├── evaluation.yaml
│   └── prompts/
│       ├── generation.yaml
│       └── evaluation.yaml
│
├── data/
│   ├── MIDAS/                 # supplied source JSON files
│   ├── PETCI/                 # supplied source JSON file
│   └── processed/             # generated JSONL and normalization reports
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py
│   ├── logging_config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── adapters.py
│   │   ├── normalize.py
│   │   └── validate.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── generation.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── builder.py
│   │
│   ├── generate.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── judge.py
│   │   ├── rubric.py
│   │   └── metrics.py
│   │
│   ├── evaluate.py
│   ├── analyze.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── output.py
│       └── reproducibility.py
│
├── slurm/
│   ├── generate_array.sh
│   └── evaluate.sh
│
├── scripts/
│   ├── discover_checkpoints.py
│   ├── prepare_data.py
│   └── validate_results.py
│
├── tests/
│   ├── test_data.py
│   ├── test_prompts.py
│   ├── test_generation.py
│   ├── test_evaluation.py
│   └── test_aggregation.py
│
├── results/
│   ├── generations/
│   ├── evaluations/
│   ├── analysis/
│   └── plots/
│
└── logs/
```

Create directories automatically when needed.

Do not commit the original source datasets in `data/MIDAS/` or `data/PETCI/`, downloaded model weights, API credentials, or large generated result files to Git. The normalized files and reports in `data/processed/` are intentionally kept under version control.

---

# 12. Command-Line Interface

Provide a simple CLI for the entire workflow.

The commands should be usable both locally and on the university cluster.

Suggested interface:

### Discover checkpoints

```bash
python scripts/discover_checkpoints.py
```

Lists available repository revisions and relevant metadata.

### Prepare datasets

```bash
python normalize_datasets.py
```

Loads the supplied MIDAS and PETCI JSON files and writes normalized JSONL plus summary/exclusion reports under `data/processed/`.

### Run one checkpoint locally

```bash
python -m src.generate --checkpoint-index 0
```

Useful for debugging and small-scale testing.

### Run the complete generation pipeline locally

```bash
python -m src.generate --all
```

This is optional and should not be the recommended way to run the full experiment on the cluster.

### Evaluate generations

```bash
python -m src.evaluate
```

Scores all available generated explanations.

### Analyze results

```bash
python -m src.analyze
```

Generates aggregate statistics, CSV files, and plots.

### Validate experiment completeness

```bash
python scripts/validate_results.py
```

Checks whether all expected checkpoint-idiom combinations have been generated and evaluated.

---

# 13. Configuration Management

Keep experiment settings outside the Python source code.

At minimum, support:

- Model repository.
- Checkpoint revisions.
- Checkpoint metadata.
- Dataset paths.
- Dataset column mappings.
- Languages.
- Prompt templates.
- Few-shot examples.
- Generation parameters.
- Evaluation model.
- Evaluation rubric.
- Output directories.
- Slurm resource settings.

The project should be configurable without requiring source-code modifications.

Use a central experiment configuration that references the other configuration files.

Validate configuration files at startup and provide helpful error messages.

---

# 14. Reproducibility and Experiment Tracking

Every generation and evaluation record must contain sufficient metadata to reproduce or audit the experiment.

Record:

- Experiment ID.
- Model repository.
- Exact model revision.
- Training step.
- Training token count.
- Dataset name and version.
- Idiom ID.
- Language.
- Prompt template version.
- Generation parameters.
- Evaluator model and version.
- Evaluation rubric version.
- Execution timestamp.
- Generation status.
- Evaluation status.

Use a fixed experiment seed where relevant.

Avoid overwriting existing results.

If a configuration changes, assign a new experiment ID or configuration version.

Do not combine results from different prompt versions or generation configurations without explicitly identifying the difference.

---

# 15. Error Handling and Reliability

The pipeline must be robust enough to run unattended on a university cluster.

Implement:

### Model loading

- Validate that the checkpoint revision exists.
- Detect unavailable GPU resources.
- Detect unsupported precision.
- Provide clear error messages for insufficient GPU memory.
- Avoid unnecessary model downloads.

### Generation

- Process records in batches.
- Save results incrementally.
- Support resuming interrupted jobs.
- Log individual record failures.
- Avoid duplicate outputs.
- Handle empty model completions.

### Evaluation

- Validate LLM judge responses.
- Retry transient API failures.
- Log malformed responses.
- Avoid silently assigning default scores.
- Support resuming partially completed evaluations.

### Analysis

- Detect missing checkpoints.
- Detect incomplete datasets.
- Report sample counts.
- Exclude unsuccessful records from averages while reporting their counts separately.
- Never silently treat missing scores as zero.

---

# 16. Testing Requirements

Write unit tests for the main components.

Tests should not require downloading OLMo or accessing a GPU.

Use mocked models and small synthetic datasets.

Test:

1. Dataset loading.
2. Dataset normalization.
3. Missing-value detection.
4. Prompt construction.
5. Prompt consistency across checkpoints.
6. Checkpoint selection.
7. JSONL serialization.
8. Resume functionality.
9. Evaluation score validation.
10. Aggregation calculations.
11. Handling missing checkpoint results.
12. Slurm task-to-checkpoint mapping.

Include a small synthetic dataset for development.

Do not include fabricated results in the actual research outputs.

Provide a CPU-only smoke test that exercises the pipeline using a mock model.

---

# 17. Documentation Requirements

The README should document:

1. Research objective.
2. Experimental methodology.
3. Repository structure.
4. Environment setup.
5. Dataset configuration.
6. Checkpoint discovery and configuration.
7. Prompt customization.
8. Local testing.
9. Slurm configuration.
10. Job submission and monitoring.
11. Evaluation configuration.
12. Statistical analysis.
13. Output formats.
14. Common troubleshooting steps.

Include commands that users can copy and execute.

Clearly distinguish between commands that work locally and commands that require the university cluster.

Document any assumptions that still require user input.

---

# 18. Implementation Priorities

Build the project incrementally, in the following order.

## Phase 1: Project foundation

- Create repository structure.
- Set up Python environment.
- Implement configuration loading.
- Implement logging.
- Implement dataset adapters and normalization.
- Create synthetic test data.

## Phase 2: Model inference

- Implement Hugging Face model loading.
- Implement checkpoint selection.
- Implement prompt construction.
- Implement batched generation.
- Implement JSONL output.
- Implement resume functionality.

## Phase 3: Local validation

- Test one checkpoint.
- Test a small subset of idioms.
- Validate output extraction.
- Check GPU memory consumption.
- Verify that prompts are identical across checkpoints.

## Phase 4: Slurm integration

- Implement the job array script.
- Map task IDs to checkpoint revisions.
- Configure output and error logs.
- Test a single task.
- Test a small array.
- Validate concurrent execution.

## Phase 5: Evaluation

- Implement the LLM judge.
- Implement score validation.
- Implement evaluation output.
- Implement optional BERTScore.
- Support human evaluation exports.

## Phase 6: Analysis

- Calculate aggregate scores.
- Generate plots.
- Calculate confidence intervals.
- Compare languages.
- Compare training progression.
- Export final CSV files.

## Phase 7: Final validation

- Validate experiment completeness.
- Check for missing records.
- Check for duplicate records.
- Verify reproducibility metadata.
- Run all tests.
- Update documentation.

---

# 19. Important Research Considerations

The implementation should support the following research principles.

### A. Same idioms across checkpoints

Every checkpoint must be evaluated on exactly the same idioms.

This is essential for meaningful comparisons.

### B. Same prompts across checkpoints

Do not change prompts as training progresses.

### C. Same generation parameters

Use consistent decoding settings.

### D. Multilingual evaluation

Report results separately by language.

Do not assume that performance differences between languages are exclusively caused by differences in idiomatic knowledge.

General language proficiency and language representation in pretraining may also contribute.

### E. Reference explanations are not necessarily exhaustive

A generated explanation can be correct even if it uses different words or includes a reasonable nuance not present in the reference.

The evaluator must assess meaning rather than exact matching.

### F. Checkpoint comparisons

All checkpoints must belong to the same training run.

Account for changes in training stages and data mixtures when interpreting results.

### G. Satisfactory performance

Do not arbitrarily define a satisfactory score in the code.

Allow the researcher to choose a threshold after examining the rubric and results.

The analysis should make it possible to identify when a model first reaches a selected performance threshold and whether it maintains that performance at subsequent checkpoints.

---

# 20. Expected Final Deliverables

At completion, the repository should provide:

### Data

- Normalized MIDAS dataset.
- Normalized PETCI dataset.
- Dataset statistics.

### Generation

- Eight checkpoint-specific JSONL files.
- Full model and prompt metadata.
- Generation logs.
- Error reports.

### Evaluation

- Eight checkpoint-specific evaluation files.
- Individual 0–4 scores.
- Evaluator reasoning.
- Optional human evaluation exports.
- Optional BERTScore results.

### Analysis

- Master results CSV.
- Average scores per checkpoint.
- Average scores per language.
- Average scores per checkpoint-language combination.
- Confidence intervals where appropriate.
- Training progression plot.
- Language comparison plots.
- Score distribution plots.
- Experiment completeness report.

### Documentation

- Complete README.
- Environment setup instructions.
- Dataset configuration instructions.
- Cluster setup instructions.
- Slurm submission instructions.
- Reproducibility instructions.

---

# 21. Instructions to the Coding Agent

Build the entire repository from scratch using this document as the primary specification.

Do not stop after generating a project skeleton. Implement the actual working pipeline.

However, do not fabricate information or make assumptions that could invalidate the experiment.

In particular:

1. Use the observed MIDAS and PETCI JSON schemas documented above; preserve the PETCI Chinese-idiom-to-English-explanation task distinction.
2. Do not invent Hugging Face checkpoint revisions. Discover and verify them.
3. Do not assume a particular university cluster or GPU partition.
4. Do not hardcode API credentials.
5. Do not automatically download all checkpoints.
6. Do not automatically submit Slurm jobs.
7. Do not silently discard failed records.
8. Do not introduce unnecessary dependencies.
9. Do not change experimental parameters between checkpoints.
10. Do not use instruction-tuned models for the primary generation experiment.

If essential information is missing, implement the configurable interface and document precisely what the user needs to provide.

Prioritize a working, testable pipeline over unnecessary abstractions.

The final product should allow the researcher to:

1. Supply the MIDAS and PETCI datasets.
2. Configure eight verified OLMo-2-7B checkpoints.
3. Configure target languages and few-shot examples.
4. Run a local smoke test.
5. Submit an eight-task Slurm job array.
6. Collect all generated explanations.
7. Evaluate the explanations.
8. Generate final statistical summaries and visualizations.

**The central goal is a reproducible experiment that measures idiomatic comprehension across the training progression of OLMo-2-7B.**