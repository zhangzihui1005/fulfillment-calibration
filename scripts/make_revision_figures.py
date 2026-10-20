"""Rebuild figures from frozen aggregates; no fitting and no interval re-estimation.
Figure 1 is intentionally neither read for editing nor rewritten.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'evidence';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8.0,'axes.labelsize':8.0,'xtick.labelsize':7.0,'ytick.labelsize':7.5,'legend.fontsize':7.0,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':180})
def blank(w=3.45,h=2.7):
 f=plt.figure(figsize=(w,h));a=f.add_subplot(111);a.spines[['top','right']].set_visible(False);return f,a
def save(f,name):
 f.tight_layout(pad=.7)
 for ext in ['pdf','svg','png']:f.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',pad_inches=.025)
 plt.close(f)
p=pd.read_csv(D/'e17/manuscript_primary_pairs_both_scores.csv')
f,a=blank(3.45,3.0)
for aa,bb,off,mk in [('Frozen','Raw',-.20,'^'),('Expanding','Frozen',0,'s'),('Expanding','Raw',.20,'o')]:
 s=p[(p.A==aa)&(p.B==bb)].sort_values('Origin');y=np.arange(4)+off;x=s.Brier_Delta.to_numpy()
 a.errorbar(x,y,xerr=[x-s.Brier_Low.to_numpy(),s.Brier_High.to_numpy()-x],fmt=mk,markersize=4,capsize=2,linewidth=1,label=aa+' minus '+bb)
a.axvline(0,linestyle=':',linewidth=.8);a.set_yticks(range(4),['O1','O2','O3','O4']);a.invert_yaxis();a.set_xlabel('Paired Brier difference');a.legend(loc='upper left',bbox_to_anchor=(0,1.31),frameon=False);a.ticklabel_format(axis='x',style='sci',scilimits=(-3,-3),useMathText=True);save(f,'figure2_primary')
c=pd.read_csv(D/'protocols/calendar_manuscript_comparison_table.csv');c=c[(c.Origin=='O2')&(c.Period=='Full')].sort_values('Version')
f,a=blank(3.45,2.6);x=c.BrierDelta.to_numpy();y=np.arange(4)
a.errorbar(x,y,xerr=[x-c.BrierCI_low.to_numpy(),c.BrierCI_high.to_numpy()-x],fmt='o',markersize=4,capsize=2,linewidth=1);a.axvline(0,linestyle=':',linewidth=.8);a.set_yticks(y,['S0: numerical','S1: cyclical','S2: one-hot','S3: omitted']);a.invert_yaxis();a.set_xlabel('Paired Brier difference');a.ticklabel_format(axis='x',style='sci',scilimits=(-3,-3),useMathText=True);save(f,'figure3_calendar')
f,a=blank(3.45,2.9)
for aa,b,off,mk,label in [('Rolling60','Expanding',-.13,'o','Rolling-60 minus Expanding'),('Approval60','Rolling60',.13,'s','Approval-60 minus Rolling-60')]:
 s=p[(p.A==aa)&(p.B==b)].sort_values('Origin');x=s.Brier_Delta.to_numpy();y=np.arange(4)+off;a.errorbar(x,y,xerr=[x-s.Brier_Low.to_numpy(),s.Brier_High.to_numpy()-x],fmt=mk,markersize=4,capsize=2,linewidth=1,label=label)
a.axvline(0,linestyle=':',linewidth=.8);a.set_yticks(range(4),['O1','O2','O3','O4']);a.invert_yaxis();a.set_xlabel('Paired Brier difference');a.legend(loc='upper left',bbox_to_anchor=(0,1.25),frameon=False);a.ticklabel_format(axis='x',style='sci',scilimits=(-3,-3),useMathText=True);save(f,'figure4_windows')
w=pd.read_csv(D/'r2/weekly_trajectory_with_frozen.csv',parse_dates=['Start'])
for o in ['O1','O2','O3','O4']:
 s=w[w.Origin==o];f,a=blank(3.45,2.95)
 for k,label,ls,mk in [('ActualRate','Realized (retrospective)','--','o'),('RawMean','Raw','-','s'),('FrozenMean','Frozen','-.','v'),('ExpandingMean','Expanding','-','^'),('PoolRate','Admitted-pool rate',':','D')]:
  a.plot(s.Start,100*s[k],linestyle=ls,marker=mk,markersize=3,linewidth=1,label=label)
 a.set_ylabel('Probability / frequency (%)');a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=4));a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));a.set_xlabel('Evaluation-week start');a.set_title(o,loc='right',fontsize=9);a.legend(frameon=False,fontsize=6.6,loc='upper left',bbox_to_anchor=(0,1.34),ncol=2);save(f,f'figure5_{o}_trajectory')
 f,a=blank(3.45,2.3)
 for key,label,mk in [('ExpandingIntercept','Expanding','o'),('FrozenIntercept','Frozen','s')]:a.plot(s.Start,s[key],marker=mk,markersize=3,linewidth=1,label=label)
 a.axhline(0,linestyle=':',linewidth=.8);a.set_ylabel('Applied intercept (weighted mean)');a.set_xlabel('Evaluation-week start');a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=4));a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));a.set_title(o,loc='left',fontsize=9);a.legend(frameon=False);save(f,f'supp_{o}_intercept')
 f,a=blank(3.45,2.5)
 for key,label,mk in [('Frozen_Raw_Brier','Frozen minus Raw','^'),('Expanding_Frozen_Brier','Expanding minus Frozen','s'),('Expanding_Raw_Brier','Expanding minus Raw','o')]:a.plot(s.Start,s[key],marker=mk,markersize=3,linewidth=1,label=label)
 a.axhline(0,linestyle=':',linewidth=.8);a.set_ylabel('Segment Brier difference');a.set_xlabel('Evaluation-week start');a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=4));a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));a.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True);a.set_title(o,loc='right',fontsize=9);a.legend(frameon=False,fontsize=6.6,loc='upper left',bbox_to_anchor=(0,1.3));save(f,f'supp_{o}_update_losses')
r=pd.read_csv(D/'e17/reliability_bins_with_intervals.csv')
for o in ['O1','O2','O3','O4']:
 for zoom in [False,True]:
  f,a=blank(3.45,3.1)
  for m,mk in [('Raw','o'),('Expanding','s')]:
   s=r[(r.Origin==o)&(r.Method==m)&(r.N>0)]
   if zoom:s=s[s.MeanPred<=.2]
   # Hollow markers flag n < 50. The rule affects display only.
   for sparse in [False,True]:
    z=s[s.N.lt(50) if sparse else s.N.ge(50)]
    if z.empty:continue
    a.errorbar(z.MeanPred,z.ObservedRate,yerr=[np.maximum(0,z.ObservedRate-z.RateCI_low),np.maximum(0,z.RateCI_high-z.ObservedRate)],fmt=mk,fillstyle='none' if sparse else 'full',markersize=3.4,label=m+(' (n < 50)' if sparse else ''),capsize=2,linewidth=.8)
    if sparse:
     for _,v in z.iterrows():
      right=v.MeanPred>(.15 if zoom else .75);top=v.ObservedRate>(.25 if zoom else .85)
      a.annotate('n='+str(int(v.N)),(v.MeanPred,v.ObservedRate),xytext=(-8 if right else 5,(-12 if m=='Raw' else -25) if top else (8 if m=='Raw' else 20)),ha='right' if right else 'left',textcoords='offset points',fontsize=5.7)
  bound=.2 if zoom else 1
  a.plot([0,bound],[0,bound],linestyle=':',linewidth=.8,label='Equality');a.set_xlim(0,bound);a.set_ylim(0,.32 if zoom else 1);a.set_xlabel('Mean predicted probability in bin');a.set_ylabel('Observed event fraction');a.set_title(o+(' / low-probability view' if zoom else ' / full range'),loc='left',fontsize=8.5);a.legend(frameon=False,loc='upper left',fontsize=6.4);save(f,f'figure6_{o}_reliability'+('_zoom' if zoom else ''))
 f,a=blank(3.45,2.4)
 for m,mk in [('Raw','o'),('Expanding','s')]:
  s=r[(r.Origin==o)&(r.Method==m)];cent=(s.Left+s.Right)/2;a.plot(cent,s.N/s.N.sum(),marker=mk,markersize=3,linewidth=.8,label=m)
 a.set_yscale('symlog',linthresh=.0001);a.set_xlabel('Fixed probability-bin midpoint');a.set_ylabel('Fraction of evaluation orders');a.legend(frameon=False);a.set_title(o,loc='left',fontsize=9);save(f,f'supp_{o}_distribution')
print('Rebuilt Figures 2–6 and supplementary diagnostics; Figure 1 was not modified.')
