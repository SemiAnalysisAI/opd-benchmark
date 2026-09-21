"""Verify the completed student's workload, scores, routing, and OPD signal."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def key(r):
    return (r['step'],r['domain'],r['index'],r['sample'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    root=args.root.resolve()
    run=root/'student'
    config=json.loads((run/'config.json').read_text())
    assert config['mode']=='mopd' and config['checkpoint'] is None
    assert config['start_step']==0 and config['steps']==40
    assert config['prompts']==128 and config['group_size']==1
    assert config['max_tokens']==256 and config['eval_size']==512
    events=list(rows(run/'events.jsonl'))
    assert events[-1]['event']=='complete'
    assert events[-1]['optimizer_updates']==40
    metrics=list(rows(run/'metrics.jsonl'))
    assert [r['step'] for r in metrics]==list(range(1,41))
    assert all(r['samples']==128 and r['accepted_policy_lag']==0 for r in metrics)
    manifest=json.loads((run/'data-manifest.json').read_text())
    dataset={}
    for domain,prefix in [('countdown','countdown4'),('graph_color','graph12')]:
        for split in ('train','dev'):
            name=f'{prefix}-{split}.jsonl'
            raw=gzip.decompress((root/'research/data'/f'{name}.gz').read_bytes())
            assert hashlib.sha256(raw).hexdigest()==manifest[name]['sha256']
            dataset[domain,split]=[json.loads(line) for line in raw.splitlines()]
    spec=importlib.util.spec_from_file_location('retained_scorer',run/'scoring.py')
    scorer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    counts=Counter()
    correct=Counter()
    train_keys=set()
    train_signatures={}
    train_lengths={}
    eval_keys=set()
    def signature(r):
        return hashlib.sha256(json.dumps({k:r[k] for k in
            ('prompt_tokens','tokens','logprobs','response','score')},sort_keys=True).encode()).hexdigest()
    for r in rows(run/'samples.jsonl'):
        label=dataset[r['domain'],r['split']][r['index']]['label']
        assert scorer.score(r['response'],label)==r['score']
        assert 0 < len(r['tokens']) <= 256
        assert len(r['tokens'])==len(r['logprobs'])
        assert all(math.isfinite(x) for x in r['logprobs'])
        assert r['sample']==0
        k=(r['split'],r['step'],r['domain'])
        counts[k]+=1
        correct[k]+=r['score']
        if r['split']=='train':
            assert key(r) not in train_keys
            train_keys.add(key(r))
            train_signatures[key(r)]=signature(r)
            train_lengths[key(r)]=len(r['tokens'])
        else:
            assert key(r) not in eval_keys
            eval_keys.add(key(r))
    assert len(train_keys)==5120
    for step in range(40):
        for domain in ('countdown','graph_color'):
            assert counts['train',step,domain]==64
    for step in (0,10,20,30,40):
        for domain in ('countdown','graph_color'):
            assert counts['dev',step,domain]==512
    assert sum(counts.values())==10240
    assert eval_keys=={(step,domain,index,0) for step in (0,10,20,30,40)
                      for domain in ('countdown','graph_color') for index in range(512)}
    evals=list(rows(run/'evaluations.jsonl'))
    assert [e['step'] for e in evals]==[0,10,20,30,40]
    for e in evals:
        for domain in ('countdown','graph_color'):
            assert e[domain]['correct']==correct['dev',e['step'],domain]
            assert e[domain]['accuracy']==correct['dev',e['step'],domain]/512
    score_keys=set()
    advantage_signatures={}
    gap_sums=defaultdict(float)
    gap_counts=Counter()
    for r in rows(run/'teacher-scores.jsonl'):
        k=key(r)
        assert k in train_keys and k not in score_keys
        assert signature(r)==train_signatures[k]
        score_keys.add(k)
        advantage_signatures[k]=hashlib.sha256(json.dumps(r['advantages']).encode()).hexdigest()
        assert len(r['tokens'])==len(r['teacher_logprobs'])==len(r['advantages'])
        for t,s,a in zip(r['teacher_logprobs'],r['logprobs'],r['advantages']):
            assert math.isfinite(a) and math.isclose(a,t-s,rel_tol=0,abs_tol=1e-12)
            gap_sums[r['step'],r['domain']]-=a
            gap_counts[r['step'],r['domain']]+=1
    assert score_keys==train_keys
    for m in metrics:
        for domain in ('countdown','graph_color'):
            k=(m['step']-1,domain)
            assert math.isclose(m['sampled_reverse_kl_by_domain'][domain],gap_sums[k]/gap_counts[k],abs_tol=1e-10)
    learner_rows=list(rows(run/'learner-scores.jsonl'))
    assert len(learner_rows)==5120
    learner_keys={key(r) for r in learner_rows}
    assert learner_keys==train_keys
    for r in learner_rows:
        assert len(r['logprobs'])==len(r['advantages'])==train_lengths[key(r)]
        assert all(math.isfinite(x) for x in r['logprobs'])
        assert hashlib.sha256(json.dumps(r['advantages']).encode()).hexdigest()==advantage_signatures[key(r)]
    routes=json.loads((run/'teachers.json').read_text())
    selection=json.loads((root/'teacher-selection.json').read_text())
    assert routes=={d:v['checkpoint']['sampler_path'] for d,v in selection['selected'].items()}
    student_id=json.loads((run/'model-info.json').read_text())['model_id']
    assert all(student_id not in path for path in routes.values())
    report={'passed':True,'optimizer_updates':40,'training_sequences':5120,
        'training_sequences_per_domain':2560,'development_sequences':5120,
        'exact_data_hashes':True,'all_sample_scores_recomputed':True,
        'exact_teacher_minus_student_advantages':True,'balanced_routing_each_update':True,
        'learner_and_teacher_scores_cover_all_training_sequences':True,
        'fresh_student':True,'zero_accepted_policy_lag':True,
        'source_sha256':hashlib.sha256((run/'source.py').read_bytes()).hexdigest()}
    (root/'final-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
