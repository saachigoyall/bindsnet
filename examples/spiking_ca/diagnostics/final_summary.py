import json, glob, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=collections.defaultdict(dict)
for f in glob.glob("variant_*_seed*.json"):
    r=json.load(open(f)); D[r['variant']][r['seed']]=r
base=D['base']
def nm(v):
    out=[]
    for p in v.split('+'):
        if p=='base': out.append('Baseline')
        elif p.startswith('bal'): out.append(f"Balanced inhibition {round(float(p[3:])*100)}%")
        elif p.startswith('wnorm'):
            t=p[5:].split('@'); out.append(f"Weight norm target {t[0]}"+(f", speed {t[1]}" if len(t)>1 else ""))
        else: out.append({'lateral5':'Lateral inhibition 5.0','shared_theta':'Shared theta','percell_theta':'Per cell theta','percell_theta_small':'Per cell theta, small','hetero_decay':'Heterogeneous decay','percell_double':'Per cell double adaptation'}[p])
    return " + ".join(out)
rows=[]
for v,R in D.items():
    a0=np.array([R[s]['acc']['0'] for s in R]); a10=np.array([R[s]['acc']['10'] for s in R])
    sh=[s for s in R if s in base]; d=np.array([R[s]['acc']['10']-base[s]['acc']['10'] for s in sh])
    h=np.mean([R[s]['hid_per_unit']['10'] for s in R])
    rows.append((v,len(R),a0.mean(),a10.mean(),a10.std(ddof=1)/np.sqrt(len(R)),d.mean(),d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else 0,h))
rows.sort(key=lambda r:-r[3])
print(f"{'variant':<34}{'n':>3}{'step0':>7}{'step10':>8}{'SE':>7}{'vs base':>9}{'SE':>7}{'hid10':>7}")
for r in rows: print(f"{nm(r[0]):<34}{r[1]:>3}{r[2]:7.3f}{r[3]:8.3f}{r[4]:7.3f}{r[5]:+9.3f}{r[6]:7.3f}{r[7]:7.2f}")
rows=[r for r in rows if r[1]>=4]; rows.sort(key=lambda r:r[3])
fig,ax=plt.subplots(figsize=(9.5,0.36*len(rows)+1.8),dpi=140)
for i,r in enumerate(rows):
    col="#6b7280" if r[0]=='base' else ("#d9822b" if '+' in r[0] else "#2a6fb0")
    ax.plot([r[2],r[3]],[i,i],color="#cbd5e1",lw=2,zorder=1)
    ax.scatter([r[2]],[i],s=36,facecolor="white",edgecolor=col,lw=1.6,zorder=3)
    ax.errorbar([r[3]],[i],xerr=[r[4]],fmt='o',color=col,ms=7,lw=1.4,zorder=4,capsize=3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{nm(r[0])} (n={r[1]})" for r in rows],fontsize=9,color="#1f2933")
ax.axvline(0.65,color="#6b7280",ls="--",lw=1); ax.axvline(0.82,color="#6b7280",ls=":",lw=1)
ax.text(0.652,len(rows)-0.3,"near chance",color="#6b7280",fontsize=9); ax.text(0.822,len(rows)-0.3,"input layer",color="#6b7280",fontsize=9)
ax.set_xlim(0.5,0.88); ax.set_ylim(-0.7,len(rows)+0.2); ax.set_xlabel("Hidden layer decoder accuracy",color="#1f2933")
ax.set_title("Step 0 (open) to step 10 (filled, mean ± SE). Orange = combined, grey = baseline",loc="left",fontsize=10.5,color="#1f2933")
ax.spines[['top','right']].set_visible(False); ax.grid(axis='x',color="#e5e7eb",lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("variants_figure_final.png")
