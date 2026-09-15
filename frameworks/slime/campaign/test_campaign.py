"""Manual CPU contracts for the isolated campaign, without changes to upstream CI."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import campaign
from scoring import score


def main():
    import numpy
    assert numpy.__version__ == '1.26.4'
    import scipy
    assert scipy.__version__ == '1.15.3'
    root = Path(__file__).resolve().parent
    labels = [json.dumps({'domain':'countdown','numbers':[1,2,3,4],'target':10}),
              json.dumps({'domain':'countdown','numbers':[1,2,3,4],'target':11})]
    assert [score('<answer>1+2+3+4</answer>', x) for x in labels] == [1.0, 0.0]
    original = SimpleNamespace(rm_url='unchanged', reward_key='task_score')
    domains = ['countdown', 'graph_color']
    os.environ['MOPD_TEACHER_URLS'] = json.dumps({d: f'http://{d}/generate' for d in domains})
    selected_urls = []
    async def fake_teacher(args, sample, **kwargs):
        selected_urls.append(args.rm_url)
        return {'meta_info': {'completion_tokens': 0, 'input_token_logprobs': [[None,1],[-1.0,2],[-2.0,3]]}}
    samples = []
    for domain in domains:
        sample = SimpleNamespace(metadata={'domain':domain, 'campaign_evaluation':False},
            response='<answer>1+2+3+4</answer>', label=labels[0], tokens=[1,2,3],
            response_length=2, index=0, weight_versions=['1'], reward=None)
        sample.get_reward_value = lambda args, sample=sample: sample.reward[args.reward_key]
        with patch.object(campaign.opd, 'reward_func', fake_teacher):
            sample.reward = asyncio.run(campaign.reward(original, sample))
        samples.append(sample)
    raw, processed = campaign.postprocess(original, samples)
    assert raw == [1,1] and processed == [0,0]
    assert all(torch.equal(s.teacher_log_probs, torch.tensor([-1.,-2.])) for s in samples)
    assert selected_urls == ['http://countdown/generate','http://graph_color/generate']
    assert original.rm_url == 'unchanged' and original.reward_key == 'task_score'
    from examples.train_infer_mismatch_helper.mis import compute_mis_weights
    metric_args = SimpleNamespace(use_tis=False, tis_lower_bound=0.5, tis_upper_bound=2.0,
                                 rs_lower_bound=None, rs_upper_bound=None)
    mask = [torch.ones(2)]
    weights, returned_masks, metrics = compute_mis_weights(metric_args,
        train_log_probs=[torch.tensor([-1.,-2.])], rollout_log_probs=[torch.tensor([-1.1,-2.1])],
        loss_masks=mask)
    assert weights is None and returned_masks is mask and metrics
    samples[0].metadata['campaign_evaluation'] = True
    assert asyncio.run(campaign.reward(original,samples[0])) == {'task_score':1.0}
    counts = {}
    for prefix, domain in [('countdown4','countdown'),('graph12','graph_color')]:
        for split, n in [('train',10000),('dev',512)]:
            path = root/'data'/f'{prefix}-{split}.jsonl'
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            assert len(rows) == n and all(r['metadata']['domain'] == domain for r in rows)
            assert all([m['role'] for m in r['prompt']] == ['system','user'] for r in rows)
            counts[path.name] = n
    print(json.dumps({'native_postprocess': 'passed', 'teacher_routing':'passed', 'zero_training_reward':'passed', 'datasets':counts}))


if __name__ == '__main__':
    main()
