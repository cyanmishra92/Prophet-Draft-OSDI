# ASPLOS'26 \- Summer cycle Paper \#1116 Reviews and Comments

Paper \#1116 Prophet: Neural Expert Prediction for Efficient Mixture-of-Experts Inference

# Review \#1116A

## Reviewer expertise

1. This is not my area, I know little about the topic

## Reviewer confidence

2. I understood the main idea but missed many details

## Paper summary

This paper tackles the problem of MoE prefetching by using a neural predictor called Prophet, which itself is implemented as a transformer model.  Prophet uses the current context of which experts have been active for recent layers, and then makes a prediction for a few layers down the line of which experts will be needed then.  This lead time serves as an opportunity to prefetch the predicted experts in GPU memory, hopefully avoiding any  I/O-related stalls.

## Significance of the problem

3. Important and emerging. Big opportunity for significant gains

## Novelty of the solution

3. Significant, encouraging others to build on results

## Correctness

3. The paper has no factual mistakes

## Writing quality

3. Well-written

## Related work

5. Comprehensive coverage, well done\!

## Robustness of the evaluation

3. Well-executed and supports claims

## Advancement in the core ASPLOS disciplines

1. Systems / Operating Systems

## Strengths

Please make note of my expertise score, which is low for this topic.

Prophet is a software-only enhancement, which they claim would be plug-and-play into existing MoE applications.  This is different from most ASPLOS topics, which typically have a hardware architecture component, which could take years to implement in real life.

## Weaknesses

Please make note of my expertise score, which is low for this topic.

The authors tuned Prophet to use the latest 3 layers of history to predict 3 layers into the future, stating that this optimized latency and accuracy.  If future systems aren't able to load the experts within the 3 layer deadline, then further out predictions could need to be made, which would lower Prophet's accuracy, or increase its complexity, both of which make it a less attractive solution.

## Comments for authors

The word "Prophet" is frequently misspelled as "Prohet."

## Overall Merit

2. Leaning reject (Generally negative, requires a major overhaul)

## Ranking relative to other papers assigned to you

1. Below 40%

# Review \#1116B

## Reviewer expertise

2. I read papers in this area but do not publish/work on these topics

## Reviewer confidence

3. I understood the paper but missed some details

## Paper summary

The paper introduces Prophet, a prefetching system to tackle the issue of large offloading latency in MoE models: expert loading. The key insight is that expert usage follows a heavy-tailed distribution (a few experts are extremely popular) and exhibits strong temporal and cross-layer correlations (the choice of an expert in an early layer helps predict the choice in a later layer).  Prophet includes a lightweight transformer model to analyze a token's recent routing history to predict which experts will be needed in subsequent layers. The predictor's confidence scores are used to make placemwnt decisions in the GPU L1 and L2 caches. Evaluation shows that Prophet improves performance by 1.5x-3.2x over existing state-of-the-art methods.

## Significance of the problem

2. Important but extensively studied. Little opportunity for significant gains

## Novelty of the solution

3. Significant, encouraging others to build on results

## Correctness

2. The paper has partially correct claims

## Writing quality

3. Well-written

## Related work

3. Many related works are missing

## Robustness of the evaluation

2. Substantial but misaligned with claims

## Advancement in the core ASPLOS disciplines

1. Systems / Operating Systems

## Strengths

* Plug-and-play system with no required model changes  
* Interesting statistical analysis to identify of batch-level optimization opportunities, cross-layer correlations and underlying structures in expert usage  
* Novel neural predictor design to leverage multi-layer routing history for expert prefetching  
* Evaluation uses diverse set of models with different scaling challenges

## Weaknesses

* The claim that this is the first paper to look at cross-layer correlations is over-stated as there are prior works that have exploited a similar observation. Comparison with these papers is not present.  
* Insights are empirical and it is unclear how broadly they will generalize to other models (with more activated experts) and datasets  
* The system was tested on the same datasets from which the training traces were collected  
* The cost and latency of the neural predictor could be significant, and these overheads are not discussed.

## Comments for authors

This is an interesting paper that addresses a critical bottleneck in MoE inference. The information-theoretic analysis provides a theoretical foundation for the predictive approach, and the results are impressive. However, the claims appear to be too strong, with missing comparison to related work and some flaws in evaluation methodology.

**Cross-Layer Prediction**

The paper claims to be the "first system to systematically exploit dependencies between routing decisions across transformer layers." While Prophet's use of neural networks is novel, the core insight of using cross-layer information has been explored in prior work. For instance, models like FATE \[14\] use cross-layer gates for predictive scheduling. Similarly MoE-Infinity \[52\] traces activations across layers. AdapMoe (not cited in paper, reference below) also does cross-layer prefetching with smart caching.

It would strengthen the paper to provide a more direct comparison, highlighting precisely how Prophet's multi-layer context window and transformer-based prediction mechanism differ from and improve upon these earlier heuristic or single-layer lookahead approaches.

Shuzhang Zhong, Ling Liang, Yuan Wang, Runsheng Wang, Ru Huang, and Meng Li. 2024\. Adapmoe: Adaptive sensitivity-based expert gating and management for efficient moe inference. In Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design. 1–9

**Generalizability**

The information-theoretic analysis provides a good motivation, but the findings are largely empirical, which raises questions about broader applicability. For example, the paper demonstrates these patterns on models activating up to 8 experts. It would be valuable to discuss how you expect these patterns—specifically the power-law distribution and temporal correlations—to hold for emerging architectures that activate a much larger number of experts (e.g., top-12 or higher). Furthermore, how might these routing structures change on more specialized or out-of-domain datasets (e.g., source code, medical literature) compared to the general-purpose corpora used?

**Evaluation Methodology**

The system was tested on the same datasets (Natural Questions, IMDb, CNN/DM) from which the predictor's training traces were collected. This approach doesn't demonstrate the predictor's ability to generalize to unseen data distributions. To make a stronger case for the robustness of the system, consider adding an experiment with a hold-out dataset. For example, training the predictor on traces from two of the datasets and then evaluating the end-to-end system performance on the third would provide more powerful evidence of generalization.

Analysis of the Neural Predictor's Overhead:

The paper would benefit from a deeper discussion of the costs of the neural predictor. For example, what is the one-time cost to collect the necessary traces and train the predictor for a new multi-billion parameter MoE model? This includes both the computational resources and time required. How does the predictor's inference latency scale with factors like sequence length and batch size? While the current overhead is small, understanding its scaling behavior is important for assessing its viability in future, more demanding scenarios. A brief discussion on the sensitivity of the predictor's performance to its own hyperparameters (e.g., context window size c, prediction horizon h) would add valuable practical insight for others looking to implement this system.

Questions:

1. Can you describe the MoE models used for the evaluation? How many experts are activated? How do your observations scale with different number of activated experts?  
2. Where is the predictor executed? Are GPU resources reserved for it? How does the predictor latency scale with sequence length and batch size?

## Overall Merit

2. Leaning reject (Generally negative, requires a major overhaul)

## Ranking relative to other papers assigned to you

1. Below 40%

# Review \#1116C

## Reviewer expertise

3. I published/worked in the area but might miss the most recent related work

## Reviewer confidence

3. I understood the paper but missed some details

## Paper summary

The paper introduces Prophet, an MoE inference system with neural network predicting expert activation. By predicting the following layers' expert activation, Prophet could prefetch expert parameters efficiently. Evaluation shows Prophet delivers both latency and memory usage improvement.

## Significance of the problem

3. Important and emerging. Big opportunity for significant gains

## Novelty of the solution

2. Low conceptual novelty

## Correctness

3. The paper has no factual mistakes

## Writing quality

2. Reasonable. Would benefit from revision

## Related work

4. Fairly comprehensive; I would add a bit more

## Robustness of the evaluation

2. Substantial but misaligned with claims

## Advancement in the core ASPLOS disciplines

1. Systems / Operating Systems

## Strengths

(1) Addresses the important problem of MoE inference under resource-constrained conditions. Improving efficiency with limited GPU resources can substantially lower the deployment barrier for MoE models.

(2) Introduces a novel idea of improving predictor accuracy with a neural network method.

(3) Provides a clear discussion of existing MoE offloading approaches and articulates the unique contributions of the proposed method.

## Weaknesses

(1) Although the paper clearly explains its progress over existing expert-weight offloading approaches, it lacks discussion and experimental comparison with two critical research directions: (a) KV cache offloading the authors should quantify the memory breakdown of MoE inference to analyze the relative contributions of expert weights vs. KV cache, and clarify whether expert offloading alone sufficiently addresses usability concerns; (b) CPU/GPU co-execution—prior work such as MoELightning and kTransformers has demonstrated the benefits of exploiting CPU compute resources in addition to memory, whereas this paper restricts computation to GPUs, which is a significant limitation.

(2) The scalability of the mixture-of-predictors approach is questionable. In real-world serving scenarios, it is difficult to partition user requests into clean task categories as assumed in benchmarks. The paper does not analyze semantic routing approaches, and the evaluation omits realistic mixed-task workloads (e.g., ShareGPT-style traces).

(3) The evaluation baselines are incomplete. Comparisons should include systems such as MoELightning and kTransformers that exploit CPU/GPU cooperation. The paper should also discuss why such approaches are not adopted here, or alternatively, how the proposed techniques could complement them.

## Comments for authors

\*\*Significance: The paper addresses a meaningful and emerging problem.

\*\*Novelty: I remain unconvinced about the generalizability of the predictor. Clarifying its applicability beyond the evaluated settings (e.g. manually selected dataset) would strengthen the contribution.

\*\*Related work: The discussion is fairly comprehensive but omits an important line of research on GPU/CPU co-execution.  A deeper comparison with such approaches is needed.

\*\*Robustness of evaluation: In addition to missing comparisons with CPU/GPU co-execution systems, the evaluation lacks analysis of KV cache memory consumption, which may be a critical factor in constrained deployments.

The paper presents promising ideas, but the lack of discussion and experimental comparison against two key techniques—KV cache offloading and GPU/CPU co-execution—limits its impact. I encourage the authors to strengthen the analysis along these dimensions to make the work more convincing.

## Overall Merit

2. Leaning reject (Generally negative, requires a major overhaul)

## Ranking relative to other papers assigned to you

2. Top 30%

# Review \#1116D

## Reviewer expertise

1. This is not my area, I know little about the topic

## Reviewer confidence

3. I understood the paper but missed some details

## Significance of the problem

3. Important and emerging. Big opportunity for significant gains

## Novelty of the solution

3. Significant, encouraging others to build on results

## Correctness

3. The paper has no factual mistakes

## Writing quality

3. Well-written

## Related work

1. I am not up-to-date with the current literature

## Robustness of the evaluation

3. Well-executed and supports claims

## Advancement in the core ASPLOS disciplines

1. Systems / Operating Systems

## Comments for authors

Thanks for submitting your paper to ASPLOS\!

The paper is well-written and the idea of using a learning predictor to accelerate a MoE model is interesting. However, as always with learning-based predictors, the main challenge is convincing the reader that the predictor is not overfitted to a very specific set of traces.

My main concern is that the generality of the results presented in this paper is unclear. The paper does not convince the reader that the observations that are made in S3 and the experimental results are applicable to other datasets. The authors analyzed 3 datasets that are then used to both train and test the predictor. A much more convincing approach would be to run the predictor on a set of different datasets.

It'd be interesting to compare (or at least cite) your work to other learning-based cache eviction policies like LRB from Learning Relaxed Belady for Content Distribution Network Caching, NSDI'20.

## Overall Merit

2. Leaning reject (Generally negative, requires a major overhaul)

## Ranking relative to other papers assigned to you

1. Below 40%
