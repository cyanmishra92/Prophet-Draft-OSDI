# Mutual Information & Expert Clustering Analysis Plan

**Goal:** Provide theoretical validation that Prophet learns **structured routing patterns** (model topology) rather than random correlations or dataset-specific patterns.

---

## 1. Mutual Information (MI) Analysis

### A. Cross-Layer Mutual Information

**Question:** How much information do expert selections at layer `i` provide about layer `i+1`?

**Hypothesis:** High MI between adjacent layers indicates structured routing patterns.

**Analysis:**
```python
MI(Expert_layer_i, Expert_layer_i+1)
```

**What it proves:**
- High MI → Expert routing has inherent structure (not random)
- Prophet can exploit this structure for prediction
- Validates the cross-layer attention mechanism

**Visualizations:**
1. Heatmap: MI matrix for all layer pairs (12×12)
2. Line plot: MI vs layer distance (adjacent, +2, +3, etc.)
3. Bar chart: Average MI per layer

---

### B. Cross-Domain Mutual Information

**Question:** Do expert routing patterns share mutual information across different domains?

**Hypothesis:** If Prophet learns topology (not data), MI should be non-zero across domains.

**Analysis:**
```python
# For each layer
MI(Expert_routing_domain_A, Expert_routing_domain_B)
```

**Experiments:**
- Q&A (SQuAD) vs Sentiment (IMDB)
- News (CNN) vs Wiki (WikiText)
- All 7 cross-domain pairs from OSDI experiments

**What it proves:**
- Non-zero cross-domain MI → Shared routing structure
- Higher MI → More similar routing topology
- Validates cross-domain generalization results

**Visualizations:**
1. Heatmap: Cross-domain MI matrix (6×6 domains)
2. Bar chart: MI for each OSDI experiment pair
3. Scatter: Cross-domain MI vs retention rate (correlation!)

---

### C. Predictor-Actual Mutual Information

**Question:** How much information does Prophet's prediction share with ground truth?

**Hypothesis:** High MI indicates Prophet captures the routing structure.

**Analysis:**
```python
# Compare predicted vs actual expert selections
MI(Prophet_predictions, Ground_truth_routing)
```

**Baselines:**
- Random predictor: MI ≈ 0
- Perfect predictor: MI = H(routing) (entropy)
- Prophet: Should be significantly > random

**What it proves:**
- High MI → Prophet learns meaningful patterns
- MI across domains → Prophet learns generalizable structure
- MI vs accuracy correlation → Information is predictive

**Visualizations:**
1. Bar chart: MI for Prophet vs Random vs Perfect
2. Line plot: MI across training epochs (learning curve)
3. Heatmap: MI per layer per domain

---

## 2. Expert Clustering & Correlation Analysis

### A. Expert Co-Activation Clustering

**Question:** Which experts are frequently activated together?

**Analysis:**
```python
# Create co-activation matrix
CoActivation[i,j] = P(expert_i AND expert_j both active)

# Cluster experts based on co-activation patterns
clusters = hierarchical_clustering(CoActivation)
```

**What it reveals:**
- Expert specialization patterns
- Redundant vs complementary experts
- Which experts work together

**Visualizations:**
1. Dendrogram: Hierarchical clustering of 128 experts
2. Heatmap: Co-activation matrix (128×128)
3. Network graph: Experts as nodes, co-activation as edges

---

### B. Cross-Layer Expert Correlation

**Question:** Do specific experts at layer `i` predict specific experts at layer `i+1`?

**Analysis:**
```python
# Correlation matrix
Correlation[expert_i, expert_j] = corr(
    activation_layer_i[expert_i],
    activation_layer_i+1[expert_j]
)
```

**What it reveals:**
- Layer-to-layer routing pathways
- Expert dependency chains
- Why cross-layer attention helps

**Visualizations:**
1. Sankey diagram: Expert flows between layers
2. Heatmap: Layer i → Layer i+1 correlation matrix
3. Graph: High-correlation expert pathways (top 10%)

---

### C. Domain-Specific Expert Specialization

**Question:** Do certain experts specialize in certain domains?

**Analysis:**
```python
# Expert activation frequency per domain
activation_freq[domain][expert] = count(expert used in domain) / total

# Compute specialization score
specialization[expert] = entropy(activation_freq_across_domains)
```

**What it reveals:**
- High specialization → Domain-specific experts
- Low specialization → General-purpose experts
- Distribution validates or refutes domain contamination

**Visualizations:**
1. Heatmap: Expert activation frequency per domain (128 experts × 6 domains)
2. Histogram: Specialization scores (entropy distribution)
3. Bar chart: Top 20 most specialized vs most general experts

---

### D. Temporal Routing Patterns

**Question:** Do routing patterns show temporal structure within sequences?

**Analysis:**
```python
# Analyze expert transitions across time steps
transition_matrix[expert_i, expert_j] = P(expert_j | expert_i at t+1)

# Compute temporal autocorrelation
autocorr = correlation(expert_seq[t], expert_seq[t+k])
```

**What it reveals:**
- Temporal dependencies in routing
- Why context window matters
- Sequential structure in expert usage

**Visualizations:**
1. Transition matrix heatmap (128×128)
2. Autocorrelation plot (lag vs correlation)
3. Markov chain visualization (top transitions)

---

## 3. Key Results We Want to Show

### For OSDI Paper:

**Main Claims:**
1. **Prophet learns topology, not data**
   - Evidence: High cross-domain MI (>0.5 bits)
   - Evidence: Expert clustering is domain-invariant

2. **Expert routing has structured patterns**
   - Evidence: High cross-layer MI (>1.5 bits)
   - Evidence: Clear expert co-activation clusters

3. **Prophet captures this structure**
   - Evidence: MI(Prophet, Ground_truth) >> MI(Random, Ground_truth)
   - Evidence: MI correlates with accuracy (r > 0.8)

4. **Structure is generalizable**
   - Evidence: Cross-domain MI > 0
   - Evidence: Expert clusters consistent across domains

---

## 4. Analysis Implementation Plan

### Phase 1: Data Preparation
- [ ] Load all 37,200 traces from `robust_traces.pkl`
- [ ] Organize by layer, domain, sample
- [ ] Create lookup tables for fast access

### Phase 2: MI Calculations
- [ ] Implement entropy calculation: `H(X)`
- [ ] Implement conditional entropy: `H(X|Y)`
- [ ] Implement mutual information: `MI(X,Y) = H(X) - H(X|Y)`
- [ ] Optimize for 128 experts × 12 layers

### Phase 3: Clustering Analysis
- [ ] Compute co-activation matrix
- [ ] Run hierarchical clustering
- [ ] Compute expert correlation matrices
- [ ] Identify expert specialization patterns

### Phase 4: Cross-Domain Analysis
- [ ] Load cross-domain experiment data
- [ ] Compute cross-domain MI for all pairs
- [ ] Correlate MI with retention rates
- [ ] Validate domain-invariance hypothesis

### Phase 5: Predictor Analysis
- [ ] Load Prophet predictions
- [ ] Compute MI(Prophet, Ground_truth)
- [ ] Compare against baselines
- [ ] Analyze per-layer and per-domain

### Phase 6: Visualization
- [ ] Generate all planned visualizations
- [ ] Export data to CSV/TEX
- [ ] Create publication-ready figures

---

## 5. Expected Computational Cost

**Data Size:**
- 37,200 traces × 12 layers = ~450K data points
- Each trace: 128-dim expert distribution
- Total: ~58M expert selection events

**MI Computation:**
- Cross-layer: 12 × 12 = 144 calculations
- Cross-domain: 6 × 6 = 36 calculations
- Predictor-actual: 12 layers × 6 domains = 72 calculations
- **Total: ~250 MI calculations**

**Clustering:**
- Co-activation matrix: 128 × 128 = 16K cells
- Hierarchical clustering: O(n² log n) ≈ fast enough
- Correlation matrices: Similar cost

**Estimated Runtime:** 10-30 minutes on single CPU

---

## 6. Output Structure

```
evaluation/mi_clustering_results/
├── mutual_information/
│   ├── cross_layer_mi.{csv,tex,pdf,png}
│   ├── cross_domain_mi.{csv,tex,pdf,png}
│   ├── predictor_mi.{csv,tex,pdf,png}
│   └── mi_vs_retention.{csv,tex,pdf,png}
│
├── clustering/
│   ├── expert_coactivation.{csv,tex,pdf,png}
│   ├── expert_dendrogram.{pdf,png}
│   ├── cross_layer_correlation.{csv,tex,pdf,png}
│   └── domain_specialization.{csv,tex,pdf,png}
│
├── temporal/
│   ├── transition_matrix.{csv,tex,pdf,png}
│   ├── autocorrelation.{csv,tex,pdf,png}
│   └── routing_chains.{pdf,png}
│
└── summary/
    ├── MI_CLUSTERING_SUMMARY.md
    ├── key_findings.json
    └── all_results_table.tex
```

---

## 7. Integration with OSDI Submission

### New Section: "Theoretical Validation"

**Section 5: Information-Theoretic Analysis**

5.1 Expert Routing Has Structure
- Cross-layer MI: X.X bits (Figure X)
- Expert clustering reveals Y clusters (Figure Y)

5.2 Prophet Learns This Structure
- MI(Prophet, Ground_truth) = Z bits
- Z/max_MI = W% of theoretical maximum

5.3 Structure is Domain-Invariant
- Cross-domain MI > 0 for all pairs
- Expert clusters consistent (correlation = V)

5.4 Validation of Cross-Domain Results
- MI correlates with retention (r = 0.XX)
- Explains why Prophet generalizes

**New Figures (2-3 recommended):**
- Figure X: Cross-layer and cross-domain MI heatmaps
- Figure Y: Expert clustering dendrogram + specialization
- Figure Z: MI vs retention rate scatter (key validation)

---

## 8. Questions for Discussion

1. **Scope:** Should we analyze all 37K traces or sample for faster iteration?
2. **Domains:** Focus on 6 OSDI domains or include all available datasets?
3. **Prophet predictions:** Do we have saved predictions or need to re-run?
4. **Visualizations:** Which are most important for main paper vs appendix?
5. **Baselines:** What random/heuristic baselines to compare against?

---

**Next Steps:**
1. Review this plan and adjust scope
2. Implement MI calculation utilities
3. Run cross-layer MI analysis first (fastest validation)
4. Iterate on visualizations
5. Expand to full analysis

**Estimated Timeline:**
- MI utilities: 1-2 hours
- Cross-layer analysis: 30 min
- Cross-domain analysis: 1 hour
- Clustering analysis: 1-2 hours
- Visualizations: 2-3 hours
- **Total: ~6-9 hours of work**

---

**Let's start! Which analysis should we tackle first?**
