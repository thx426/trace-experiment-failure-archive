"""Render measured results; run after demo.py with and without --tied-readout."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
data = json.loads((ROOT/'results_tied/results.json').read_text())
untied = json.loads((ROOT/'results/results.json').read_text())
arms = ['discrete', 'naive_latent', 'soft_reembed', 'trained_endpoint', 'trained_intermediate']
names = ['显式符号反馈', '未经循环训练的隐状态反馈', '概率加权后重新嵌入', '循环训练：仅末轮监督', '循环训练：逐轮监督']
hops = [1, 2, 4, 8, 16]
plt.rcParams.update({'font.family': 'Arial Unicode MS', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.unicode_minus': False, 'figure.facecolor': 'white'})

def stats(dataset, arm, field='accuracy_by_hop'):
    x = np.array([r['metrics'][arm][field] for r in dataset['runs']])*100
    return x.mean(0), x.std(0, ddof=1)

rows = []
for variant, dataset in [('tied', data), ('untied', untied)]:
    for arm in arms:
        mean, sd = stats(dataset, arm)
        for k in range(len(mean)):
            rows.append({'readout':variant, 'arm':arm, 'hop':k+1,
                         'accuracy_percent':float(mean[k]), 'seed_sd_percentage_points':float(sd[k])})
with (ROOT/'summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={'width_ratios':[1.65,1]})
matrix = np.array([stats(data, arm)[0][np.array(hops)-1] for arm in arms])
im = axes[0].imshow(matrix, cmap='Blues', vmin=0, vmax=100, aspect='auto')
axes[0].set_yticks(range(5), names)
axes[0].set_xticks(range(5), hops)
axes[0].set_xlabel('需要执行的跳数 k（循环适配深度为 4）')
axes[0].set_title('A  测试准确率 %：共享输入嵌入与输出权重', loc='left', pad=14)
for i,arm in enumerate(arms):
    m, sd=stats(data, arm)
    for j,k in enumerate(hops):
        label = f'{m[k-1]:.2f}' if 99.9 < m[k-1] < 100 else f'{m[k-1]:.1f}'
        axes[0].text(j,i,label,ha='center',va='center',color='white' if m[k-1]>60 else '#10263d')
for sp in axes[0].spines.values(): sp.set_visible(False)
keys = ['no_intervention','predicted_embedding','gold_embedding','wrong_embedding']
labels = ['原隐状态', '模型预测符号的嵌入', '正确符号的嵌入（oracle）', '错误符号的嵌入（对照）']
values = [np.mean([r['intervention_hop2'][key] for r in data['runs']])*100 for key in keys]
axes[1].barh(range(4), values, color=['#a25520','#176da0','#6694ad','#b9c1c6'])
axes[1].set_yticks(range(4), labels)
axes[1].invert_yaxis()
axes[1].set_xlim(0,113)
axes[1].set_xticks([0,50,100])
axes[1].set_xlabel('第二跳准确率 %')
axes[1].set_title('B  仅替换第一轮结束后的状态',loc='left',pad=14)
for i,v in enumerate(values): axes[1].text(v+1.8,i,f'{v:.2f}',va='center')
fig.suptitle('能读出中间答案，不等于下一轮能直接使用它',fontsize=17,y=1.02)
fig.text(0.02,-0.025,'本地机制实验；每个条件 3 个种子 × 2,048 道测试题。基础模型仅单步训练；非预训练 LLM 基准。',fontsize=11)
fig.tight_layout(w_pad=2.4)
fig.savefig(ROOT/'summary.png',dpi=180,bbox_inches='tight')
fig.savefig(ROOT/'summary.svg',bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(11.5,4.3))
colors = {'naive_latent':'#b46625','trained_endpoint':'#296e99','trained_intermediate':'#31866c'}
x=np.arange(1,17)
for dataset,style,label,color in [(data,'-','直接隐状态反馈：共享输出权重','#b46625'),
                                  (untied,'--','直接隐状态反馈：独立输出权重','#8850a3')]:
    m,s=stats(dataset,'naive_latent')
    ax.plot(x,m,style,marker='o',label=label,color=color)
    ax.fill_between(x,np.maximum(0,m-s),np.minimum(100,m+s),alpha=.12,color=color)
m,s=stats(data,'trained_endpoint')
ax.plot(x,m,'-',lw=2,color='#296e99',label='经过循环训练（末轮监督）≈ 100%')
ax.set_ylim(0,106); ax.set_xlim(1,16); ax.set_xticks(hops)
ax.set_xlabel('所需推理跳数');ax.set_ylabel('测试准确率 %')
ax.grid(axis='y',alpha=.18)
ax.legend(loc='center right',frameon=False)
ax.set_title('权重绑定减轻了未经适配的循环错误；循环训练进一步修复了接口',loc='left',pad=14)
fig.text(.08,-.005,'阴影为三个训练种子的样本标准差，不是置信区间。显式符号反馈和软重嵌入各跳均为 100%。',fontsize=10)
fig.tight_layout()
fig.savefig(ROOT/'depth_sensitivity.png',dpi=180,bbox_inches='tight')
plt.close(fig)

case = data['runs'][0]['case']
fig,ax=plt.subplots(figsize=(12,3.6))
lanes = [('正确路径',case['gold']),('显式符号',case['predictions']['discrete']),
         ('未适配循环：读出',case['predictions']['naive_latent']),
         ('已训练循环：读出',case['predictions']['trained_endpoint'])]
for y,(label,values) in enumerate(lanes):
    ax.text(-.8,y,label,ha='right',va='center',fontsize=11)
    for i,val in enumerate(values[:8]):
        good=val==case['gold'][i]
        box=FancyBboxPatch((i-.35,y-.24),.7,.48,boxstyle='round,pad=0.04',
                            fc='#dcecf5' if good else '#f6dfce',ec='none')
        ax.add_patch(box)
        ax.text(i,y,str(val),ha='center',va='center',color='#12374f' if good else '#8f4314',fontsize=13)
        if i<7: ax.text(i+.5,y,'→',ha='center',va='center',color='#6b747a')
ax.set_title(f'同一道新题：从 {case["start"]} 出发，沿关系表走 8 跳',loc='left',pad=18)
ax.set_xlim(-.6,7.5);ax.set_ylim(3.6,-.7);ax.axis('off')
fig.text(.12,.00,'隐式两行显示的是逐轮探针读出；这些数字没有作为 token 回送。案例按预先写入代码的失败条件选择。',fontsize=10)
fig.tight_layout()
fig.savefig(ROOT/'trajectory.png',dpi=180,bbox_inches='tight')
plt.close(fig)

lines=['| 方法 | 1 跳 | 2 跳 | 4 跳 | 8 跳 | 16 跳 |', '|---|---:|---:|---:|---:|---:|']
for arm,name in zip(arms,names):
    mean,sd=stats(data,arm)
    lines.append('| '+name+' | '+' | '.join(f'{mean[k-1]:.2f} ± {sd[k-1]:.2f}' for k in hops)+' |')
(ROOT/'measured_table.md').write_text('\n'.join(lines)+'\n')
print('Rendered summary.png, depth_sensitivity.png, trajectory.png, summary.csv, measured_table.md')
