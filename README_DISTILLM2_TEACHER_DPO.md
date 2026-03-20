# Teacher-Prompt DPO and `distillm2_token`

This document explains the custom teacher-prompt DPO flow implemented in this repository, with special focus on the
`TeacherPromptAlignedDPOTrainer` and the `distillm2_token` loss.

It is written to answer four practical questions:

1. What exactly is different from standard DPO?
2. What tensors are computed during training?
3. What does the `distillm2_token` loss optimize?
4. What do the logged metrics mean, especially compared with DPO?

## Files to Read First

- `src/alignment/dpo_trainer.py`
- `src/alignment/configs.py`
- `scripts/custom_dpo.py`

## High-Level Summary

Standard DPO uses one prompt per pair and optimizes a pairwise preference objective built from sequence-level
log-probability differences between the policy and a reference model.

This repository adds a teacher-prompt variant in which:

- the policy still sees the original prompt,
- the reference or teacher side can see a different prompt,
- the student and teacher share the same completion tokens,
- evaluation falls back to the original prompt so metrics remain comparable to normal DPO evaluation.

The repository also adds `distillm2_token`, which is not a standard DPO loss. Instead of optimizing a pairwise
sigmoid or hinge objective over sequence-level log-ratios, it optimizes a token-level KL mixture:

- chosen branch: forward KL from teacher to policy,
- rejected branch: reverse KL from policy to teacher.

## Class Structure

### `DPOTrainer`

`DPOTrainer` is the base trainer for normal text DPO and VLM DPO in this codebase.

Its responsibilities include:

- loading the model and reference model,
- preparing and tokenizing the dataset,
- building batches,
- computing policy and reference sequence log-probabilities,
- computing the configured DPO-style loss,
- logging metrics such as rewards, log-probs, margins, entropy, and token accuracy.

### `TeacherPromptAlignedDPOTrainer`

`TeacherPromptAlignedDPOTrainer` extends `DPOTrainer` for the special case where the teacher or reference prompt can
be different from the policy prompt.

Its main extra responsibilities are:

- detecting teacher-prompt fields in the raw dataset,
- tokenizing teacher prompts separately,
- building teacher-specific batches that reuse the same completion tokens,
- computing teacher-side reference log-probabilities from `teacher_input_ids`,
- computing token-level KL terms used by `distillm2_token`.

## Dataset Flow

### Standard DPO Flow

For normal DPO, each example contains:

- `prompt`
- `chosen`
- `rejected`

The trainer tokenizes:

- `prompt_ids`
- `chosen_ids`
- `rejected_ids`

Then it builds two sequences per example:

- `prompt_ids + chosen_ids`
- `prompt_ids + rejected_ids`

The completion mask marks only the completion tokens, so prompt tokens do not contribute to the loss.

### Teacher-Prompt Flow

For teacher-prompt training, the raw dataset may additionally contain:

- `teacher_prompt`
- `teacher_chosen_prompt`
- `teacher_rejected_prompt`

The policy side still uses the normal student prompt:

- chosen branch input: `prompt_ids + chosen_ids`
- rejected branch input: `prompt_ids + rejected_ids`

The teacher side uses separately tokenized prompt ids:

- chosen teacher input: `teacher_chosen_prompt_ids + chosen_ids`
- rejected teacher input: `teacher_rejected_prompt_ids + rejected_ids`

The key design choice is that both sides share the same completion tokens. Only the prompt prefix can differ.

## Why the Shared Completion Constraint Matters

`distillm2_token` computes token-level KL only on completion positions. To compare policy and teacher token
distributions token by token, the chosen completion length on the policy side must match the chosen completion length
on the teacher side, and the same must hold for the rejected branch.

That is why `TeacherPromptAlignedDPOTrainer` validates completion-mask lengths before computing token-level KL terms.

## Train vs Eval Prompt Behavior

This trainer intentionally behaves differently in train and eval:

- Train:
  teacher prompts can differ from the student prompt.
- Eval:
  teacher prompts are ignored and the original `prompt` is used instead.

This happens in `resolve_teacher_prompt()` inside `_prepare_dataset()`.

The reason is simple: evaluation is easier to interpret when the policy and reference are both scored against the
same original prompt rather than a custom teacher-only prompt. This keeps reward-style metrics closer to standard DPO
evaluation semantics.

## Batch Construction

### Student Batch

The student batch contains:

- `input_ids`
- `attention_mask`
- `completion_mask`

The ordering is:

- first half: chosen sequences
- second half: rejected sequences

### Teacher Batch

The teacher-aware collator adds:

- `teacher_input_ids`
- `teacher_attention_mask`
- `teacher_completion_mask`

These teacher tensors follow the same chosen-then-rejected ordering.

## Core Quantities

The trainer computes several quantities that are easy to confuse. Here is the exact meaning of each one.

### `chosen_logps` and `rejected_logps`

These are sequence-level completion log-probabilities from the current policy model.

They are computed by:

1. running the policy model on the student batch,
2. taking next-token logits,
3. converting them to token log-probabilities for the observed labels,
4. zeroing out non-completion positions,
5. summing over completion tokens.

In shorthand:

- `chosen_logps = log pi_theta(y_chosen | x_student)`
- `rejected_logps = log pi_theta(y_rejected | x_student)`

### `ref_chosen_logps` and `ref_rejected_logps`

These are sequence-level completion log-probabilities from the reference model.

For teacher-prompt training they are computed on the teacher batch:

- `ref_chosen_logps = log pi_ref(y_chosen | x_teacher_chosen)`
- `ref_rejected_logps = log pi_ref(y_rejected | x_teacher_rejected)`

During eval, because teacher prompts collapse back to the original prompt, these become closer to normal DPO
reference scores.

### `chosen_logratios` and `rejected_logratios`

These are policy-minus-reference sequence log-probability differences:

- `chosen_logratios = chosen_logps - ref_chosen_logps`
- `rejected_logratios = rejected_logps - ref_rejected_logps`

Interpretation:

- positive value: the policy assigns higher probability than the reference,
- negative value: the policy assigns lower probability than the reference.

These values are always computed, even for `distillm2_token`, because they are useful for logging and DPO-style
comparisons.

## Standard DPO Losses in This Trainer

For the standard losses such as `sigmoid`, `hinge`, `ipo`, `robust`, and others, the trainer first builds
`chosen_scores` and `rejected_scores` from the log-ratios and the configured f-divergence. Then it constructs the
loss from the difference:

- `delta_score = chosen_scores - rejected_scores`

For the default reverse-KL DPO case:

- `chosen_scores = chosen_logratios`
- `rejected_scores = rejected_logratios`

Then standard DPO-style losses use `beta` directly. Example:

- sigmoid DPO: `-logsigmoid(beta * delta_score)`
- hinge: `relu(1 - beta * delta_score)`

In these losses, `beta` is a real training parameter. It changes the objective itself.

## `distillm2_token` Loss

`distillm2_token` is different.

It does not directly optimize a pairwise objective over `chosen_logratios` and `rejected_logratios`.

Instead it computes token-level KL terms over completion tokens only.

### Chosen Branch: Forward KL

For the chosen branch, the trainer computes forward KL from the teacher distribution to the policy distribution:

- `KL(teacher || policy)`

This encourages the policy to cover modes that the teacher assigns probability to on the chosen completion.

### Rejected Branch: Reverse KL

For the rejected branch, the trainer computes reverse KL from the policy distribution to the teacher distribution:

- `KL(policy || teacher)`

This is mode-seeking relative to the teacher and is used on the rejected branch in the custom objective.

### Final Loss

The final `distillm2_token` loss is:

- `(1 - rkl_weight) * chosen_forward_kl + rkl_weight * rejected_reverse_kl`

If `rkl_weight` is unset, the trainer uses:

- `rkl_weight = 0.5`

That means the default behavior is a symmetric mix:

- 50 percent chosen forward KL,
- 50 percent rejected reverse KL.

### Role of `beta` in `distillm2_token`

In `distillm2_token`, `beta` is no longer a training-mix parameter.

It does not scale the token-KL objective. The loss is controlled by:

- `rkl_weight`
- `trust_region_alpha`
- the teacher and policy token distributions themselves

`beta` is still required to be positive because reward dashboards are logged on the standard DPO scale:

- `reward = beta * logratio`

So for `distillm2_token`:

- `rkl_weight` affects training,
- `beta` affects reward logging only.

## Trust-Region Teacher

The trainer optionally mixes two teacher distributions in log-probability space:

- the fixed reference model,
- the current model run on the teacher prompt.

This is controlled by:

- `trust_region_alpha`

Behavior:

- `trust_region_alpha = 0.0`: use only the fixed reference teacher,
- `trust_region_alpha > 0.0`: geometrically mix the fixed reference and current teacher distributions.

The mixed teacher is stop-gradient. It acts as a target distribution and is not backpropagated through as a trainable
teacher branch.

This trust-region mechanism is only enabled for `distillm2_token`.

## Logging and Metric Semantics

The trainer logs both general model metrics and DPO-style preference metrics.

### Metrics That Reflect the Actual `distillm2_token` Training Objective

These are the most faithful metrics for understanding `distillm2_token` optimization:

- `kl/chosen_forward`
- `kl/rejected_reverse`
- `teacher_mix/alpha`

Interpretation:

- lower `kl/chosen_forward` means the policy is getting closer to the teacher on chosen completions,
- lower `kl/rejected_reverse` means the policy is matching the rejected-side reverse-KL target better,
- `teacher_mix/alpha` reports the configured trust-region mix.

### DPO-Style Reward Metrics

The trainer also logs:

- `rewards/chosen`
- `rewards/rejected`
- `rewards/accuracies`
- `rewards/margins`

These are always logged on the DPO scale:

- `chosen_reward = beta * chosen_logratios`
- `rejected_reward = beta * rejected_logratios`

This is true even for `distillm2_token`.

Interpretation:

- `rewards/chosen`: how much more the policy favors the chosen completion than the reference, on the DPO scale,
- `rewards/rejected`: same idea for the rejected completion,
- `rewards/accuracies`: fraction of pairs where `chosen_reward > rejected_reward`,
- `rewards/margins`: average value of `chosen_reward - rejected_reward`.

Important caveat:

For `distillm2_token`, these reward metrics are monitoring metrics, not the optimized objective itself.

They are useful because:

- they keep dashboards comparable to DPO runs,
- they show whether the KL-based training is moving the policy in a preference-aligned direction,
- they help compare teacher-prompt distillation against normal DPO.

They should not be mistaken for the training loss.

### Other Logged Metrics

- `logps/chosen` and `logps/rejected`: raw policy completion log-probability sums,
- `logits/chosen` and `logits/rejected`: average raw logits on completion positions,
- `mean_token_accuracy`: next-token accuracy on chosen completions only,
- `entropy`: token entropy on completion positions,
- `num_tokens`: cumulative processed token count.

## DPO vs `distillm2_token`

### Similarities

- Both use the same chosen and rejected completion data.
- Both compute policy and reference sequence log-probabilities.
- Both log DPO-style rewards from `beta * logratio`.
- Both can be evaluated with the same reward dashboards.

### Differences

#### Optimization Target

Standard DPO:

- optimizes a pairwise preference objective over sequence-level scores.

`distillm2_token`:

- optimizes token-level KL matching objectives, split by branch.

#### Role of `beta`

Standard DPO:

- `beta` directly changes the loss.

`distillm2_token`:

- `beta` does not change the loss,
- `beta` only changes DPO-style reward logging.

#### Role of `rkl_weight`

Standard DPO:

- not used.

`distillm2_token`:

- controls how much weight goes to rejected reverse KL versus chosen forward KL.

#### Interpretation of `rewards/*`

Standard DPO:

- reward metrics are close to the actual training objective.

`distillm2_token`:

- reward metrics are auxiliary diagnostics on a DPO-compatible scale.

## Practical Guidance

### If You Care About Training Dynamics

Watch:

- `loss`
- `kl/chosen_forward`
- `kl/rejected_reverse`

These tell you whether the actual KL objective is improving.

### If You Care About Preference Alignment

Watch:

- `rewards/accuracies`
- `rewards/margins`
- `rewards/chosen`
- `rewards/rejected`

These tell you whether the policy is moving in a DPO-like preferred direction relative to the reference model.

### If You Compare to Plain DPO

Use:

- the same eval dataset,
- the same `beta`,
- the same reward dashboard definitions.

That makes `rewards/*` directly comparable across DPO and `distillm2_token` runs, even though the training losses are
different.

## Example Command

```bash
python scripts/custom_dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --teacher_chosen_prompt_instruction "Answer the user's request directly with the strongest possible assistant response. Prioritize correctness, helpfulness, clear task completion, and faithful instruction-following. Use careful reasoning when needed, but keep the visible answer natural, relevant, and well-structured. Avoid unsupported claims, unnecessary refusal, and irrelevant digressions." \
    --teacher_rejected_prompt_instruction "Answer the user's request directly, but under a strict quality filter that rejects weak responses. Strongly avoid factual errors, shallow reasoning, contradiction, irrelevance, vagueness, unnecessary verbosity, missing user constraints, and overconfident unsupported claims. Prefer precise, coherent, grounded, and instruction-faithful answers." \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 200 \
    --output_dir distillm2DPO_trust_region \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --loss_type distillm2_token \
    --beta 0.1 \
    --rkl_weight 0.5 \
    --trust_region_alpha 0.4
```

Interpretation of the key flags:

- `loss_type distillm2_token`: enable token-level KL training,
- `rkl_weight 0.5`: equal weighting between chosen FKL and rejected RKL,
- `trust_region_alpha 0.4`: mix 40 percent of the current teacher-prompt distribution into the fixed reference teacher target,
- `beta 0.1`: use DPO-style rewards scaled by 0.1 in logs.

## Short Takeaway

If you want to think in DPO terms:

- DPO trains on `beta * (chosen_score - rejected_score)`.

If you want to think in this repository's `distillm2_token` terms:

- training is token-level KL matching,
- teacher prompts may differ from student prompts during training,
- evaluation uses the original prompt,
- `rkl_weight` controls the loss mix,
- `beta` keeps reward dashboards on a familiar DPO scale.
