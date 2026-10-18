# Design document: category suggestions for new administrative court cases

> Follows the [ml-design-docs](https://github.com/eugeneyan/ml-design-docs) template. Section numbers are fixed: later labs revise sections by these numbers.
> Where a value is not known yet, it is written as **Assumption:** and kept until a later lab replaces it.

## 1. Overview

_TODO (Lab 1): 3–5 sentences — problem, solution, desired outcome. Not revised later._

## 2. Motivation

_TODO (Lab 1): why a system and not a notebook — who consumes the predictions, how often, what happens when they are wrong._

## 3. Success metrics

_TODO (Lab 1): three layers kept apart, each with a target or an assumption — business, model, system._

## 4. Requirements and constraints

### Functional requirements

_TODO (Lab 1): inputs, outputs, who calls, what is returned._

### Non-functional requirements

_TODO (Lab 1): one row per category, each with a number or an explicit assumption — freshness; latency and throughput; availability and fallback; reproducibility; adaptability; observability without ground truth; cost and resource ceilings._

### 4.1 In scope and out of scope

_TODO (Lab 1)._

## 5. Methodology

### 5.1 Problem statement

_TODO (Lab 1): the framing, and one rejected framing with the reason._

### 5.2 Data

_TODO (Lab 1): sources, arrival pattern, labels and their delay, serving-time vs training-time availability, and the data checklist (time axis, history, label delay, text corpus)._

### 5.3 Techniques

_TODO (Lab 1): baseline first, then the candidate; feature list with the serving-time source of every feature._

### 5.4 Experimentation and validation

_TODO (Lab 1): split strategy justified by the time axis, offline metric, how a new model is compared with the current one._

### 5.5 Human in the loop

_TODO (Lab 1): who reviews a drift alert, who approves a promotion, what an override looks like._

## 6. Implementation

### 6.1 High-level design

_TODO (Lab 1): system context, data flow (seven layers, three marks), deployment (first draft)._

### 6.2 Infra

_TODO (Lab 1): local stack and the managed AWS / GCP / Azure service per component (names from the course reference table)._

### 6.3 Performance

_TODO (Lab 1): throughput and latency budgets; batch vs online, with the reason._

### 6.4 Security

_TODO (Lab 1): where secrets live, who may call the service._

### 6.5 Data privacy

_TODO (Lab 1): personal data in the dataset, retention._

### 6.6 Monitoring and alarms

_TODO (Lab 1, made real in Lab 4): what is logged, which metrics, which threshold fires, who receives the alert._

### 6.7 Cost

_TODO (Lab 1): estimated monthly cost of the cloud column, from the providers' calculators, with assumptions._

### 6.8 Integration points

_TODO (Lab 1): upstream data owner, downstream consumer, the contract with each._

### 6.9 Risks and uncertainties

_TODO (Lab 1): named risks, one mitigation each._

## 7. Appendix

### 7.1 Alternatives

_TODO (Lab 1): at least two rejected choices with pros, cons and the reason._

### 7.2 Experiment results

_Placeholder — filled in Lab 2._

### 7.3 Performance benchmarks

_Placeholder — filled in Lab 4._

### 7.4 Milestones

_TODO (Lab 1): one sentence per later lab on what it adds._
