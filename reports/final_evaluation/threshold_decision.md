1. I evaluated **1601** distinct alert patterns using validation scores only.

2. The validation set has **1546 normal observations**. My **5% FPR cap** permits at most **77 false alarms**.

3. The largest feasible number of detected failures is **50/54**, giving **92.59% recall**.

4. Among thresholds achieving that recall within the cap, I chose **0.062125949322206** to minimize false alarms: **63**, or **4.08%** of normal observations.

5. This leaves **4 missed failures** and produces **113 alerts**. Of these alerts, **55.75% are false alarms** (precision **44.25%**).

6. Compared with the earlier maximum-F1 threshold, this catches **11 more failures** and adds **56 false alarms**.

7. Detecting more failures requires at least **102 false alarms** in this validation ranking. The next such operating point is **0.040486104533590**, detecting **51 failures** at **6.60% FPR**. That adds **1 detected failure(s)** and **39 false alarms**, so it exceeds my chosen cap.

The selected FPR can be below 5% because predictions change in discrete steps and I prefer fewer false alarms when recall is tied. The cap is a validation selection constraint, not a guarantee for future observations.
