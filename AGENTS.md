# AGENTS.md

## Project Purpose

Build a clean, reproducible deep-learning emulator for daily spatial ECH2O outputs using a **ConvLSTM or ConvGRU encoder plus a full-resolution spatial decoder**.

Primary targets:

- `tskin_am`
- `tskin_pm`
- `plc_am`
- `plc_pm`
- `soilmoisture`

The central modeling hypothesis is:

> Daily ECH2O outputs depend on an evolving, spatially heterogeneous hydroclimatic state that cannot be represented adequately by a static U-Net receiving hand-engineered lag and rolling-summary channels.

The model should therefore learn:

1. a spatially distributed temporal state from sequential forcing rasters;
2. local nonlinear responses to current conditions and static covariates;
3. contextual spatial corrections without destroying fine-scale structure.

Do not reproduce the legacy preprocessing or training code. Build a new project around an explicit data contract, validated sequences, modular experiments, and reproducible evaluation.

---

## Scientific Framing

Let:

- `X_t(s)` be dynamic forcing rasters at time `t` and location `s`;
- `S(s)` be static terrain, vegetation, soil, and climate rasters;
- `H_t(s)` be the learned spatial hidden state;
- `Y_t(s)` be the ECH2O target field.

The core state update is:

```text
H_t = RecurrentSpatialEncoder(X_t, H_{t-1})
```

The target prediction is:

```text
Y_hat_t = SpatialDecoder(H_t, X_t, S)
```

The preferred conceptual decomposition is:

```text
prediction = local response + state-aware spatial correction
```

The model must be capable of representing regime-dependent effects such as:

- energy-limited versus water-limited conditions;
- post-precipitation cooling;
- drought and dry-down;
- seasonal radiation changes;
- topographic redistribution of water;
- vegetation-mediated thermal and hydraulic responses;
- AM versus PM process differences.

---

## Dynamic Inputs

Daily forcing rasters currently include:

- precipitation;
- shortwave radiation (`srad`);
- minimum temperature (`tmin`);
- maximum temperature (`tmax`);
- minimum relative humidity (`rmin`);
- maximum relative humidity (`rmax`).

These forcings are spatially varying rasters, not bbox-level scalar values.

The recurrent encoder should consume the raw ordered daily sequence:

```text
[B, T, C_dynamic, H, W]
```

where:

- `B` = batch size;
- `T` = sequence length;
- `C_dynamic` = number of dynamic forcing channels;
- `H`, `W` = spatial dimensions.

Hand-engineered lag and rolling features may be retained as auxiliary predictors, but they must not replace the raw temporal sequence.

Existing engineered features may include:

- 1-, 2-, and 3-day lags;
- 3-, 14-, and 30-day precipitation accumulations;
- 3-, 7-, and 15-day rolling means for other forcings.

Treat these as optional decoder inputs or benchmark-model predictors.

---

## Static Inputs

Static raster channels may include:

### Terrain and hydrology

- elevation;
- slope;
- aspect-derived northness and eastness;
- topographic wetness index;
- topographic position index;
- flow accumulation;
- multiscale terrain metrics when available.

### Climate context

- climatic water deficit;
- long-term climate normals;
- aridity or energy-balance context when available.

### Vegetation

- vegetation class;
- canopy cover;
- LAI or related vegetation structure metrics;
- plant functional type;
- other vegetation statics.

### Soil

- texture;
- depth;
- hydraulic or water-holding properties;
- other soil statics.

Static tensors should have shape:

```text
[B, C_static, H, W]
```

Categorical rasters must not be treated as continuous without an explicit design decision. Use embeddings, one-hot encoding, or another documented encoding.

---

## Target Inputs and Outputs

Targets should have shape:

```text
[B, C_target, H, W]
```

with a stable channel order defined in configuration.

Recommended default target order:

```text
0: soilmoisture
1: tskin_am
2: tskin_pm
3: plc_am
4: plc_pm
```

The order must never be inferred from file-system ordering.

Masks must be carried explicitly:

```text
[B, 1 or C_target, H, W]
```

Never convert nodata to zero unless zero is physically meaningful and the mask remains available.

---

## Canonical Sample Contract

Every training sample must resolve to:

```text
site_id
bbox_id
start_date
end_date
target_date
dynamic_sequence
static_stack
current_forcing
optional_engineered_features
target_stack
valid_mask
spatial_metadata
```

Required metadata:

- coordinate reference system;
- affine transform;
- spatial resolution;
- bbox dimensions;
- site identifier;
- date range;
- target date;
- dynamic channel names;
- static channel names;
- target channel names;
- nodata conventions.

Before model training, create and validate a manifest containing one row per eligible sequence-target sample.

---

## Sequence Construction

The model requires continuous chronological sequences.

Rules:

1. Preserve site and bbox identity.
2. Preserve temporal order.
3. Never shuffle time steps within a sequence.
4. Reset recurrent state at site or bbox boundaries.
5. Do not bridge missing days silently.
6. Record sequence length and missingness.
7. Exclude or explicitly mask invalid time steps.
8. Ensure the target date is strictly aligned with the final sequence step.

Initial sequence lengths to benchmark:

- 15 days;
- 30 days;
- 60 days;
- 90 days.

Start with 30 days unless data availability or compute constraints require otherwise.

Sequence length must be configuration-driven.

---

## Missing Days and Gaps

A recurrent model must not assume that two adjacent records are consecutive days unless this is verified.

For each sample:

- verify exact daily spacing;
- record missing dates;
- reject sequences with gaps in the baseline implementation;
- add masked-gap support only after the baseline is correct.

Possible later extensions:

- elapsed-time channels;
- forcing masks;
- hidden-state reset after long gaps;
- learned missing-step embeddings.

Do not implement these extensions before establishing a correct contiguous-sequence baseline.

---

## Primary Architecture

### 1. Recurrent spatial encoder

Begin with either:

- ConvGRU; or
- ConvLSTM.

Prefer ConvGRU as the first implementation because it has fewer parameters and lower memory demand.

Input:

```text
[B, T, C_dynamic, H, W]
```

Output:

```text
hidden_state: [B, C_hidden, H, W]
```

The recurrent hidden state must preserve full spatial resolution unless an explicitly tested multiscale design is used.

Do not pool dynamic forcing fields before the first baseline is established.

### 2. Full-resolution spatial decoder

The initial decoder should avoid a classic U-Net bottleneck.

Recommended baseline:

```text
concat(
    final_hidden_state,
    current_forcing,
    static_stack,
    optional_engineered_features
)
    -> residual convolution blocks
    -> target heads
```

Use full-resolution residual blocks composed of:

- `3 x 3` convolutions;
- normalization selected explicitly;
- nonlinear activation;
- residual connections;
- no spatial pooling.

The decoder should preserve fine spatial structure.

### 3. Local response branch

Add a local branch after the recurrent baseline is working.

```text
current pixel covariates
    -> 1 x 1 convolutional MLP
    -> local prediction
```

The contextual branch predicts a correction:

```text
final_prediction = local_prediction + contextual_correction
```

This branch is intended to reproduce the strong local nonlinear behavior observed in GBMs while retaining spatially varying recurrent state.

### 4. Multi-target heads

Use either:

- one shared decoder with five output channels; or
- shared trunk plus target-specific heads.

Prefer shared trunk plus target-specific heads because the targets represent related but distinct physical responses.

Suggested heads:

```text
soilmoisture_head
tskin_am_head
tskin_pm_head
plc_am_head
plc_pm_head
```

Each head should initially be shallow.

---

## Soil Moisture as State Supervision

Soil moisture currently retains spatial structure better than the other targets. Treat this as a modeling clue.

The shared hidden state should be supervised by soil moisture, either through:

1. a soil-moisture target head;
2. an auxiliary soil-moisture loss;
3. a predicted soil-moisture field passed into the diagnostic heads.

Preferred staged design:

```text
dynamic forcing sequence
        -> ConvGRU / ConvLSTM
        -> latent spatial state
            -> soil moisture head
            -> diagnostic trunk
                -> tskin_am
                -> tskin_pm
                -> plc_am
                -> plc_pm
```

Later experiment:

```text
diagnostic heads receive predicted soil moisture
```

Never feed contemporaneous true ECH2O soil moisture into other target heads during deployment evaluation.

Teacher forcing may be tested only if the train-inference discrepancy is handled explicitly and documented.

---

## Optional Conditional Modulation

After the recurrent baseline is established, test FiLM-style conditioning.

A global or regional regime vector may be derived from the recurrent state and used to scale and shift decoder features:

```text
Z_l_conditioned = gamma(H_t) * Z_l + beta(H_t)
```

This may help the model interpret terrain differently under:

- energy-limited conditions;
- water-limited conditions;
- wet-up;
- dry-down;
- snow or snowmelt;
- high-radiation events.

Do not add FiLM before the simpler concatenation baseline has been evaluated.

---

## Benchmark Models

The project must preserve strong non-deep-learning baselines.

Required experiments:

### B0: GBM with engineered history

Purpose:

- establish the strongest tabular/local benchmark;
- quantify performance without learned recurrent state.

### B1: Static U-Net

Purpose:

- reproduce the current spatial benchmark;
- quantify smoothing and spatial-structure loss.

### B2: ConvGRU plus full-resolution decoder

Purpose:

- test whether explicit spatial memory improves performance.

### B3: ConvLSTM plus full-resolution decoder

Purpose:

- compare recurrent cell choices.

### B4: ConvGRU plus local 1 x 1 branch

Purpose:

- test whether local nonlinear response and recurrent spatial context can coexist.

### B5: ConvGRU plus soil-moisture auxiliary supervision

Purpose:

- test whether explicit hydrologic-state supervision improves Tskin and PLC.

### B6: GBM plus recurrent spatial residual

Purpose:

- retain GBM local detail;
- use the recurrent network only for state-aware spatial correction.

For B6:

```text
final_prediction = out_of_fold_gbm_prediction + recurrent_residual
```

Never train the residual model using in-sample GBM predictions.

---

## Data Splits and Leakage Control

At minimum, support:

- spatial holdout;
- temporal holdout;
- spatiotemporal holdout.

Persist split manifests to disk.

Do not regenerate splits implicitly during training.

Leakage rules:

1. A site in the spatial test set must not appear in training.
2. Future dates must not contribute to past-sequence features.
3. Rolling features must be computed causally.
4. Normalization statistics must be estimated from training data only.
5. GBM residual training must use out-of-fold predictions.
6. Adjacent overlapping windows from the same site and date range must not cross train-test boundaries.
7. Sequence extraction must not include days after the target date.

---

## Normalization

Dynamic and continuous static channels must be normalized using training-only statistics.

Store:

- mean;
- standard deviation;
- minimum and maximum if used;
- transformation type;
- channel name;
- split identifier.

Possible transformations:

- log transform for strongly skewed positive variables;
- signed log where scientifically defensible;
- clipping based on training quantiles;
- standard scaling.

Do not normalize categorical classes as continuous variables.

Target scaling should be target-specific.

All transformations must be invertible or explicitly documented.

---

## Loss Functions

Start with a transparent multi-target loss:

```text
L_total = sum_k w_k * L_k
```

where each `L_k` is a masked MAE or Huber loss for one target.

Initial recommendation:

- Huber loss for Tskin;
- MAE or Huber loss for soil moisture;
- target-appropriate robust loss for PLC.

After the baseline is stable, test spatial-structure losses:

```text
L_target = L_pixel + lambda_grad * L_gradient
```

Optional later extension:

```text
L_target = L_pixel + lambda_grad * L_gradient + lambda_lap * L_laplacian
```

Gradient and Laplacian losses must be masked correctly around nodata cells.

Do not add complex losses until plain pixel-loss baselines have been evaluated.

---

## Evaluation Metrics

Report metrics per target and overall.

Required scalar metrics:

- RMSE;
- MAE;
- bias;
- correlation;
- R-squared where meaningful;
- standard-deviation ratio;
- normalized RMSE where useful.

Spatial-structure metrics:

- observed versus predicted standard deviation;
- gradient MAE;
- Laplacian error;
- spatial correlation;
- semivariogram comparison;
- transect comparison;
- high- and low-quantile performance.

Primary smoothing diagnostic:

```text
SD_ratio = SD(prediction) / SD(observation)
```

Interpretation:

- near 1: spatial variability retained;
- below 1: smoothing;
- above 1: excessive spatial variance.

Evaluate separately by:

- site;
- season;
- year;
- target;
- climatic regime;
- moisture regime;
- energy regime;
- wet-up and dry-down conditions;
- elevation or climatic-water-deficit strata.

---

## Regime-Stratified Evaluation

Create diagnostic groups using only variables available for evaluation.

Potential grouping variables:

- precipitation percentile;
- antecedent precipitation;
- SRAD percentile;
- temperature percentile;
- humidity or VPD proxy;
- climatic water deficit;
- predicted or target soil-moisture percentile;
- season;
- days since precipitation.

At minimum, compare:

- wet / low energy;
- wet / high energy;
- dry / high energy;
- transitional dry-down;
- post-precipitation wet-up.

The purpose is to determine whether memory improves regime-dependent predictions rather than only average performance.

---

## Visualization Requirements

For every experiment, generate a standard diagnostic panel containing:

- observed target map;
- predicted target map;
- residual map;
- target versus prediction scatterplot;
- observed and predicted distributions;
- one or more spatial transects;
- gradient-magnitude maps;
- time-series plots for selected pixels or site summaries.

For recurrent models, also inspect:

- hidden-state channel maps;
- hidden-state evolution through time;
- response before and after precipitation events;
- response during high-radiation dry periods.

Do not interpret hidden channels as physical states without evidence.

---

## Repository Structure

Use a structure similar to:

```text
configs/
    data/
    model/
    experiment/

src/
    data/
        manifest.py
        sequence_dataset.py
        transforms.py
        validation.py
    models/
        convgru.py
        convlstm.py
        fullres_decoder.py
        local_branch.py
        multitask_model.py
    training/
        losses.py
        trainer.py
        checkpoints.py
    evaluation/
        metrics.py
        regime_metrics.py
        plots.py
    utils/
        config.py
        logging.py
        reproducibility.py

scripts/
    build_manifest.py
    validate_data.py
    train.py
    evaluate.py
    compare_experiments.py

tests/
    test_manifest.py
    test_sequence_alignment.py
    test_shapes.py
    test_masks.py
    test_leakage.py

artifacts/
    manifests/
    splits/
    normalization/
    checkpoints/
    predictions/
    metrics/
    figures/
```

Keep preprocessing that creates canonical model-ready tensors separate from model training.

---

## Configuration

All experiment-defining choices must live in configuration files.

Required configurable fields:

- sequence length;
- dynamic channel list;
- static channel list;
- target list and order;
- bbox dimensions;
- recurrent cell type;
- hidden channels;
- number of recurrent layers;
- decoder depth;
- decoder channels;
- local branch enabled or disabled;
- soil-moisture auxiliary head enabled or disabled;
- loss weights;
- batch size;
- learning rate;
- split manifest;
- normalization artifact;
- random seed.

The full resolved configuration must be copied into every experiment output directory.

---

## Reproducibility

Every run must record:

- git commit;
- resolved config;
- random seed;
- software environment;
- data manifest identifier;
- split identifier;
- normalization identifier;
- model parameter count;
- training duration;
- best checkpoint criterion.

Set deterministic behavior where practical and document nondeterministic GPU operations.

---

## Tests Required Before Training

Before launching a full training run, tests must verify:

1. tensor dimensions;
2. target-channel order;
3. daily sequence continuity;
4. target-date alignment;
5. static and dynamic raster alignment;
6. mask correctness;
7. nodata handling;
8. training-only normalization;
9. split isolation;
10. recurrent state reset between sites;
11. no future leakage in engineered features;
12. inverse transformation of predictions.

Create a tiny smoke-test dataset containing a few sites and dates.

The entire training and evaluation pipeline must run on the smoke-test dataset before scaling up.

---

## Implementation Order

Follow this sequence strictly.

### Phase 1: Inspect and document schemas

1. Inspect target data.
2. Inspect dynamic forcing data.
3. Inspect static raster data.
4. Identify all linking keys.
5. Identify CRS, resolution, extent, and nodata conventions.
6. Produce a schema report.

Do not write model code before completing this phase.

### Phase 2: Build the canonical manifest

1. Create one row per eligible sequence-target sample.
2. Validate daily continuity.
3. Validate spatial alignment.
4. Persist train, validation, and test splits.
5. Create the smoke-test subset.

Stop and report validation results before proceeding.

### Phase 3: Build the dataset loader

1. Load raw daily sequences.
2. Load statics.
3. Load targets and masks.
4. Apply training-derived transformations.
5. Assert all tensor shapes.
6. Test batching.

### Phase 4: Reproduce baselines

1. GBM with engineered features.
2. Static U-Net baseline.
3. Save common metrics and figures.

### Phase 5: Build recurrent baseline

1. Implement ConvGRU.
2. Add full-resolution decoder.
3. Predict all targets.
4. Compare against baselines.

### Phase 6: Add state supervision

1. Add target-specific heads.
2. Add soil-moisture auxiliary supervision.
3. Evaluate Tskin and PLC improvement.

### Phase 7: Add local-context decomposition

1. Add the 1 x 1 local branch.
2. Add contextual correction branch.
3. Compare spatial-detail metrics.

### Phase 8: Advanced experiments

Only after the above are stable:

- ConvLSTM comparison;
- FiLM conditioning;
- longer sequences;
- gradient-aware loss;
- GBM plus recurrent residual;
- multiscale recurrent states;
- scheduled sampling or autoregressive state propagation.

---

## Coding Standards

- Use clear type hints.
- Use named tensor dimensions in comments and assertions.
- Avoid hidden global state.
- Keep model classes small and composable.
- Use structured logging.
- Fail fast on alignment errors.
- Do not silently drop invalid samples.
- Do not infer channel order from filenames.
- Do not hard-code paths.
- Do not mix data preparation with training logic.
- Write tests for every transformation that changes shape, order, time alignment, or masking.

---

## Model Simplicity Rule

Do not add architectural complexity unless a specific diagnostic justifies it.

For every new component, document:

1. the observed failure mode;
2. the scientific or statistical hypothesis;
3. the architectural change;
4. the expected metric improvement;
5. the experiment that can falsify the hypothesis.

Examples:

- Add recurrent state because static U-Net errors persist during dry-down.
- Add local branch because recurrent decoder smooths fine-scale variation.
- Add soil-moisture supervision because latent state appears insufficiently hydrologic.
- Add gradient loss because predictions retain mean accuracy but lose spatial variance.

Do not add attention, transformers, graph networks, or multiscale recurrent stacks merely because they are available.

---

## First Deliverable

The first deliverable is not a trained model.

It is a validated report containing:

- source schemas;
- spatial dimensions;
- temporal coverage;
- dynamic channels;
- static channels;
- target channels;
- site and bbox identifiers;
- missing-day statistics;
- nodata statistics;
- spatial-alignment checks;
- proposed sequence lengths;
- number of eligible sequences;
- proposed train, validation, and test splits;
- unresolved data-contract issues.

After producing this report, stop and request review before implementing the recurrent architecture.

---

## Default First Recurrent Experiment

Unless diagnostics require otherwise, the first recurrent experiment should use:

```text
Encoder: ConvGRU
Sequence length: 30 days
Recurrent layers: 1
Hidden state: full spatial resolution
Decoder: full-resolution residual CNN
Pooling: none
Targets: all five outputs
Heads: shared trunk plus target-specific shallow heads
Loss: masked target-weighted Huber or MAE
Auxiliary supervision: soil moisture enabled
Local 1 x 1 branch: disabled initially
Engineered lag features: optional decoder inputs
Evaluation: spatial, temporal, and spatiotemporal holdouts
```

This experiment is intended to answer one question cleanly:

> Does explicit spatially distributed temporal memory improve ECH2O target prediction while retaining fine-scale spatial structure?
