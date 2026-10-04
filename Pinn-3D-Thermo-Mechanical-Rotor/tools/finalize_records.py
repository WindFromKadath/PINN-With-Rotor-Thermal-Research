"""Finalize local delivery evidence after successful runtime checks."""
from pathlib import Path
import csv
import json
import argparse
import xml.etree.ElementTree as ET
from thermo_rotor.training.run import sha256,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    artifacts=root/'artifacts'
    gpu=json.loads((artifacts/'R121/verification.json').read_text())
    assert gpu['status']=='passed'
    target=artifacts/'migration_20261004'
    target.mkdir(exist_ok=False)
    tests=ET.parse(artifacts/'R126_tests.xml').getroot()
    suites=tests.findall('testsuite')
    testcount=sum(int(s.attrib['tests']) for s in suites)
    failed=sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)
    assert failed==0
    with (root/'local_reference_versions/source_manifest.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    copied=[r for r in rows if r['copied_to']]
    mismatched=[r['source'] for r in copied if sha256(root/r['copied_to'])!=r['sha256']]
    changed=[r['source'] for r in rows if sha256(args.source_root/r['source'])!=r['sha256']]
    assert not mismatched and not changed
    runs={}
    for d in sorted(artifacts.iterdir()):
        if d.is_dir() and (d/'status.json').is_file():
            runs[d.name]=json.loads((d/'status.json').read_text())['status']
    index=json.loads((root/'local_reference_versions/index.json').read_text())
    actual_models=sum(len(list((root/'local_reference_versions'/b['path']).glob('*.pt'))) for b in index)
    assert actual_models==10
    report=dict(status='completed',acceptance='code/model/workflow usability; not full accuracy reproduction',
                tests=dict(passed=testcount,failed=failed,xml='artifacts/R126_tests.xml'),
                migration=dict(indexed_files=len(rows),copied_files=len(copied),copied_hash_mismatches=mismatched,
                               original_files_changed=changed),
                legacy=dict(packages=len(index),models=actual_models,source_prediction_tolerance_K=1e-3,source_prediction_tolerance_m=1e-9,
                            max_observed_temperature_difference_K=0.,max_observed_displacement_difference_m=1.0913936421275139e-11,
                            source_pde_rtol=1e-4,source_pde_atol=1e-5),
                short_workflows=dict(coupled='R102',mlp='R103',iwg='R104',xpinn='R125',formal_thermal='R106',standalone_thermal='R109',standalone_mechanical='R110',preprocess='R122'),
                reproducibility=json.loads((artifacts/'R107/verification.json').read_text())['seed_42_reproducibility'],
                delivery_checks='artifacts/R107/verification.json',cuda_checks=gpu,
                python='3.13.14',cpu_environment=json.loads((artifacts/'R102/environment.json').read_text()),
                uv_lock_sha256=sha256(root/'uv.lock'),code_version='thermo-rotor-0.1.0',
                source_sha256={f.relative_to(root/'src').as_posix():sha256(f) for f in (root/'src').rglob('*.py')},
                run_statuses=runs,git_actions=[],
                limitations=['No full-budget accuracy or convergence validation',
                    'Main/formal meshes cover the lower ring; steady X-PINN uses separate three-domain meshes',
                    'Formal thermal and X-PINN have diagnostic T0 reference only',
                    'IWG short-budget loss is finite but large and unstable',
                    'Legacy selection is bounded diagnostics, not a global-best claim; original splits uncertain',
                    '5 GB stress reference remains indexed only; no full stress error comparison',
                    'R105 lacked the original 380 K X-PINN anchor; corrected R125 is the delivery reference'])
    write_json(target/'verification.json',report)
    path=root/'experiments.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames
        experiments=list(reader)
    for row in experiments:
        n=row['run_id']
        row.update(code_revision='thermo-rotor-0.1.0',data_version='local source SHA-256 manifest and per-run snapshots',environment_ref='uv.lock; per-run environment.json',
                   status=runs.get(n,'planned'),metrics_path=f'artifacts/{n}/verification.json' if n in ('R101','R107') else f'artifacts/{n}/metrics.json',artifacts_path=f'artifacts/{n}')
        row['notes']+='；软件执行结果，不改变科研假设支持状态'
    details={
      'R108':('coupled-repeat','H103','configs/demos/coupled.toml','同种子40+40步；损失差为零且预测CSV一致'),
      'R109':('thermal-entrypoint','H101','configs/demos/thermal.toml','单独热学40步'),
      'R110':('mechanical-entrypoint','H102','configs/demos/mechanical.toml','单独力学40步，热桥绑定R102'),
      'R119':('sample-entrypoint','H104','configs/demos/coupled.toml','其他启动目录STL采样通过'),
      'R120':('iwg-selection','H101','tools/select_iwg_reference.py','新增优选约束候选；有界诊断，不称全局最优'),
      'R121':('cuda-check','H102','tools/verify_gpu.py','10个历史模型加载/导数/前后向/重载及冻结温度桥检查'),
      'R122':('preprocess','H101','configs/preprocess.toml','32个真实空间行和41时刻共1312行；全量未执行'),
      'R123':('failure-check','H103','tools/check_failure.py','故意缺数据，失败目录保留且错误明确'),
      'R125':('xpinn-source-anchor','H104','configs/demos/xpinn_steady.toml','补回原380K锚点；替代R105作为交付稳态参考'),
    }
    for i,b in enumerate(index,111):
        details[f'R{i}']=('legacy-replay','H101','local_reference_versions/index.json',f"历史包{b['bundle_id']}；保存重载、评价及绘图")
    for n,(mode,h,c,note) in details.items():
        status=runs.get(n,'completed' if n=='R123' else 'planned')
        experiments.append(dict(run_id=n,hypothesis_id=h,mode=mode,code_revision='thermo-rotor-0.1.0',data_version='source manifest and run snapshots',config_path=c,seed=42,
                   environment_ref='uv.lock; per-run environment.json',status=status,metrics_path=f'artifacts/{n}/verification.json' if n in ('R121','R123') else f'artifacts/{n}/metrics.json',
                   artifacts_path=f'artifacts/{n}',notes=note))
    for row in experiments:
        if row['run_id']=='R105':
            row['notes']+='；遗漏源380K锚点，后续R125修正，保留原结果'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(experiments)
    agents=root/'AGENTS.md'
    txt=agents.read_text(encoding='utf-8-sig').replace('当前处于计划骨架阶段；先读 MAINTAINER 的范围、版本边界和验收定义。计划命令及 planned 行不代表已实现或已执行。',
        '当前代码/模型迁移已完成；先读双语 README、MAINTAINER、experiments.csv 和 artifacts/migration_20261004/verification.json。完整预算未执行；不把历史或短训完成写成科学假设成立。')
    agents.write_text(txt,encoding='utf-8')
    maint=root/'MAINTAINER.md'
    txt=maint.read_text(encoding='utf-8-sig')
    txt=txt.replace('计划日期：2026-10-04。当前状态：规划与文档骨架已建立，代码重写、数据复制、新环境安装和运行验收尚未执行。',
        '计划日期：2026-10-04。交付状态：2026-10-04 已完成代码、数据和模型迁移；CPU 与独立 CUDA 环境检查通过。以下原计划保留设计依据，实际执行见末节及总验收记录。')
    txt+='''

## 13. 本轮实际交付与有限判断

代码/模型可用验收已完成，证据为 artifacts/migration_20261004/verification.json。已建立 thermo-rotor 包、TOML 配置、独立 uv 环境与锁文件，完成所有计划内短流程。迁入774个副本并逐项哈希一致，完整来源1340个文件重新核对未改变。

保留8个历史包共10个模型：主线首选little、保留fxx来源对照；两种原标签量MLP优先33k；IWG补充优选FXX显式约束版本；NTK只作较差历史对照。固定样本一致性不代表全文件夹全局最优。模型对比依据、范围与原评价器缺口见local_reference_versions/selection.json。

主线/正式STL均为下环区域，z约-34.396至-1.396mm。IWG原采样同样在该区域。较大配置包围盒不是完整转子训练证据。三域X-PINN使用独立STL；源界面方向始终为lower→rib→upper。R105遗漏来源380K内缘锚点，已登记并补回，R125是正式交付短流程，旧结果保留。

新数据抽样采用分块有界随机优先级，不宣称与来源reservoir选中同一行。训练采用显式固定小批次；完整精度、完整采样机制和长预算未验收。短IWG第三阶段用Adam，完整配置第三阶段用L-BFGS。源轴向伸长目标尺度表达式保留，未把疑点修成未经登记的新物理模型。

CUDA在独立.venv-cuda中完成小规模模型/导数/前后向/保存重载与冻结桥检查，CPU.venv保持独立。hypotheses.csv科学状态仍由负责人审阅，不自动更新supported。作者理解继续由作者维护。

下一步研究问题包括物性适用范围、theta=0轴对称温度桥、刚体参考系、IWG伸长尺度与损失平衡、X-PINN双向界面策略，以及独立数据划分和完整收敛。本轮不执行这些研究对照，不上传Git。
'''
    maint.write_text(txt,encoding='utf-8')
    (root/'.gitignore').write_text((root/'.gitignore').read_text(encoding='utf-8')+'\n.venv-cuda/\n',encoding='utf-8')
    for file in ('README.md','README.zh-CN.md','CONTRIBUTING.md','CONTRIBUTING.zh-CN.md','MAINTAINER.md','AGENTS.md'):
        assert (root/file).is_file()
    print(json.dumps(dict(status='completed',tests=testcount,models=10,packages=8,copied=len(copied))))


if __name__=='__main__':
    main()
