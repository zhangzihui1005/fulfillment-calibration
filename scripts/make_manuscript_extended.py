"""Rebuild the extended manuscript tables and shared-axis risk-reversal figure."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out",type=Path,required=True)
a=ap.parse_args()
EX=Path(__file__).resolve().parents[1]/"evidence"
L=a.out;L.mkdir(parents=True,exist_ok=True)
for folder in ["figures","tables"]:(L/folder).mkdir(exist_ok=True)
def write(p,s):(L/p).write_text(s,encoding="utf-8")
d=pd.read_csv(EX/"E18/decomposition.csv").query('Backbone=="Core" and Reference=="Frozen"').sort_values("Origin")
tex=r"""\begin{table}[t]\centering
\caption{Realized correction magnitude and error alignment for Expanding minus Frozen, S1/LR. The terms use the full evaluation cohort. Seven-day intervals for the net difference appear in Table~\ref{tab:triplets}; term-specific intervals are in Supplement S10.}
\label{tab:alignment}\footnotesize\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrr}\toprule Origin & $P$ & $A$ & $P-A$\\\midrule
"""
for _,r in d.iterrows():tex+=f"{r.Origin} & {r.P:.8f} & {r.A:.8f} & {r.Delta:+.8f}"+r"\\"+"\n"
tex+=r"\bottomrule\end{tabular}\end{table}"+"\n";write("tables/alignment.tex",tex)
scores=pd.read_csv(EX/"E19/absolute_scores.csv");effects=pd.read_csv(EX/"E19/paired_effects.csv")
tex=r"""\begin{table*}[t]\centering
\caption{All four prespecified simulation environments under Closed admission, full 120-day evaluation. Absolute expected Brier scores and paired Expanding minus Frozen effects are means over 100 sequences. Intervals are marginal $t_{99}$ intervals for mean effects. Probability MSE is $\operatorname{mean}[(q-\pi)^2]$. All information policies and five methods are in Supplement S11.}
\label{tab:simulation}\footnotesize\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrr}\toprule Environment & Raw & Frozen & Expanding & Expanding$-$Frozen [95\% interval] & Expanding MSE\\\midrule
"""
for sc in ["Correct","FixedBias","Reversal","Shape"]:
    q=scores.query('Scenario==@sc and Policy=="Closed" and Period=="Full" and Metric=="ExpectedBrier"').set_index("Method")
    e=effects.query('Scenario==@sc and Policy=="Closed" and Period=="Full" and Metric=="ExpectedBrier" and Updated=="Expanding" and Reference=="Frozen"').iloc[0]
    mse=scores.query('Scenario==@sc and Policy=="Closed" and Period=="Full" and Metric=="ProbabilityMSE" and Method=="Expanding"').iloc[0].Mean
    tex+=f"{sc} & {q.loc['Raw','Mean']:.8f} & {q.loc['Frozen','Mean']:.8f} & {q.loc['Expanding','Mean']:.8f} & {e.Mean:+.8f} [{e.CI_low:+.8f}, {e.CI_high:+.8f}] & {mse:.8f}"+r"\\"+"\n"
tex+=r"\bottomrule\end{tabular}\end{table*}"+"\n";write("tables/simulation_overview.tex",tex)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none",
                    "axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,3,figsize=(8,2.7),layout="constrained",sharey=True)
for ax,pol in zip(axes,["Immediate","Observed","Closed"]):
    q=effects.query('Scenario=="Reversal" and Policy==@pol and Updated=="Expanding" and Reference=="Frozen" and Metric=="ExpectedBrier"').set_index("Period").loc[["Phase1","Phase2","Full"]]
    for x,(period,color) in enumerate(zip(["Phase1","Phase2","Full"],["#F58518","#54A24B","#4C78A8"])):
        r=q.loc[period];ax.errorbar(x,r.Mean,yerr=[[r.Mean-r.CI_low],[r.CI_high-r.Mean]],fmt="o",color=color,capsize=4)
    ax.axhline(0,color="#555",lw=.8);ax.set_xticks(range(3),["High risk","Low risk","Full"])
    ax.set_title(pol);ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,-3))
axes[0].set_ylabel("Expected Brier: Expanding - Frozen")
fig.savefig(L/"figures/simulation_reversal.pdf",bbox_inches="tight")
fig.savefig(L/"figures/simulation_reversal.png",bbox_inches="tight",dpi=220);plt.close(fig)
