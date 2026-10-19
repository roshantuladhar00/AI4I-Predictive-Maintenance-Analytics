# Final evaluation summary

I selected **weighted_voting** in the modeling stage and retained its **uncalibrated** score representation.

## Why I used this threshold

I prioritized failure recall subject to an agreed **5% validation FPR cap**. The complete validation search selected **0.062125949322206**, detecting **50/54 failures** with **63 false alarms**. I saved the model hash, feature rules, threshold, and evaluation protocol before final-test scoring.

## Final test results

- Observations: **2000**, including **68 failures**.

- Detected failures: **61**; missed failures: **7**.

- Recall: **89.71%**.

- False alarms: **88**; FPR among normal observations: **4.55%**.

- Precision: **40.94%**; false alarms among alerts: **59.06%**.

- F1: **0.5622**; AP: **0.8703**; ROC AUC: **0.9730**.

- The observed test FPR is at or below the 5% cap.

- The test FPR 95% Wilson interval is **3.71%–5.58%**.

## Limits and next steps

- The test was previously included in early EDA, so this is not a fully blinded external evaluation.

- The split is random and does not establish generalization to later time periods or different machines.

- The dataset is synthetic; interval calculations assume independent rows.

- The 5% FPR cap does not limit the false-alarm share of alerts to 5%. Alert workload still needs operational review.

- I did not refit, recalibrate, change the threshold, or select another model after test scoring.

- I next need an inference interface using the saved feature preparation and policy, followed by independent operational validation.
