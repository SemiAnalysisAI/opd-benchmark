"""No-network regressions: `python test_campaign.py` (importing shared.scoring needs reasoning-gym==0.1.25)."""
import contextlib, io, json, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx

from campaign import TEACHER_DEFAULTS, Campaign, DOMAINS
from shared import recipe
from lifecycle import STOP_USD, check, cleanup, write


def done(value=None):
    return NS(result=lambda: value)


class Session:
    """The one training session. `weight` and `optimizer` count the updates each reflects."""
    def __init__(self):
        self.updates = self.weight = self.optimizer = 0
        self.states, self.scored, self.restored = {}, [], []

    def get_info(self):
        return NS(model_id='0', model_dump=lambda: {})

    def save_weights_for_sampler(self, name):
        return done(NS(path=name))

    def save_state(self, name):
        self.states[name] = (self.weight, self.optimizer)
        return done(NS(path=name))

    def load_state_with_optimizer(self, path):
        self.restored.append(path)
        self.weight, self.optimizer = self.states.get(path, (self.weight, self.optimizer))
        return done()

    def load_state(self, path):
        self.weight = self.states[path][0]
        self.scored.append(self.weight)
        return done()

    def forward(self, data, **kwargs):
        outputs = [{'logprobs': NS(data=[-0.5] * len(d.loss_fn_inputs['target_tokens'].data))} for d in data]
        return done(NS(metrics={}, loss_fn_outputs=outputs))

    forward_backward = forward

    def optim_step(self, params):
        assert self.weight == self.optimizer, 'Learned without the matching optimizer state'
        self.learning_rate = params.learning_rate
        self.updates, self.weight, self.optimizer = self.updates + 1, self.weight + 1, self.optimizer + 1
        return done(NS(metrics={}))


def campaign(root, dev_score=lambda index: 1, **settings):
    """A campaign on fake rows. Training rewards alternate 0, 1 within each group."""
    rows = {d: {'train': [{'_index': i} for i in range(recipe.TRAIN_EXAMPLES_PER_DOMAIN)],
                'dev': [{'_index': i} for i in range(recipe.EVAL_EXAMPLES[d])]} for d in DOMAINS}
    c = Campaign(root, None, rows, **settings)
    c.batches = []
    c.service = NS(sessions=[], create_sampling_client=lambda **kwargs: NS(close=lambda: None))

    def create_training_client(**kwargs):
        assert not c.service.sessions, 'The backend permits only one session'
        c.service.sessions.append(Session())
        return c.service.sessions[0]

    def sample_batch(out, work, step, split, group=1, temperature=1):
        c.batches.append((out.name, step, split, [row['_index'] for _, row in work]))
        return [[{'domain': d, 'index': row['_index'], 'sample': j, 'prompt_tokens': [1, 2, 3], 'tokens': [4, 5],
                  'logprobs': [-1.0, -1.0], 'score': j % 2 if split == 'train' else dev_score(row['_index']),
                  'stop_reason': 'stop'} for j in range(group)] for d, row in work]

    c.service.create_training_client, c.sample_batch = create_training_client, sample_batch
    return c


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_full_campaign_on_one_session(self):
        c = campaign(self.root)
        for domain in (*DOMAINS, None):
            c.train_stage(domain)
        session, = c.service.sessions
        # Teachers stop at their first evaluation (25 updates each). The student starts from the
        # pristine base and scores every update with both frozen 25-update teachers.
        first, updates = TEACHER_DEFAULTS['eval_every'], recipe.UPDATES
        self.assertEqual((session.updates, session.weight, session.optimizer), (2 * first + updates, updates, updates))
        self.assertEqual(session.scored, [first] * (2 * updates))
        self.assertEqual(sum(m['samples'] for m in lines(self.root / 'student/metrics.jsonl')), 2560)
        # Prior spend counts against the budget.
        write(self.root / 'resources.json', {'budget_start_unix': time.time(), 'prior_spend_estimate_usd': STOP_USD + 1})
        self.assertRaises(RuntimeError, check, self.root)

    def test_resume_keeps_update_and_prompt_offset(self):
        c = campaign(self.root, resume_plan={'caesar_cipher': {'step': 10, 'restore_reference': 'cross_job://a/step-10'}})
        c.train_stage('caesar_cipher')
        self.assertEqual(c.service.sessions[0].restored[0], 'cross_job://a/step-10')
        self.assertEqual(c.batches[0], ('teacher-caesar_cipher', 10, 'train', c.order['caesar_cipher'][320:352]))
        self.assertEqual(lines(self.root / 'teacher-caesar_cipher/checkpoints.jsonl')[-1]['prompt_offset'], 800)
        self.assertEqual(json.loads((self.root / 'teacher-caesar_cipher/complete.json').read_text())['updates'], 25)

    def test_teacher_settings_override(self):
        # 85/126 correct is below the 0.70 target but within the 0.03 tolerance. The earlier
        # run used 32-prompt batches, so the plan carries offset 640 rather than 20 * 16.
        settings = {'prompts': 16, 'learning_rate': 5e-5, 'eval_every': 5, 'max_steps': 10000, 'target_tolerance': .03}
        plan = {'caesar_cipher': {'step': 20, 'prompt_offset': 640, 'restore_reference': 'cross_job://a/step-20'}}
        c = campaign(self.root, lambda index: float(index < 85), teacher_settings=settings, resume_plan=plan)
        c.train_stage('caesar_cipher')
        train = [batch[3] for batch in c.batches if batch[2] == 'train']
        self.assertEqual((train[0], train[-1]), (c.order['caesar_cipher'][640:656], c.order['caesar_cipher'][704:720]))
        self.assertEqual((c.service.sessions[0].learning_rate, c.service.sessions[0].updates), (5e-5, 5))
        self.assertEqual([m['samples'] for m in lines(self.root / 'teacher-caesar_cipher/metrics.jsonl')], [128] * 5)

    def test_prompts_render_with_thinking_and_fit_context(self):
        calls = []
        rows = {d: {s: [{'_index': 0, 'prompt': []}] for s in ('train', 'dev')} for d in DOMAINS}
        tokenizer = NS(apply_chat_template=lambda prompt, **kwargs: calls.append(kwargs) or [0] * 200)
        Campaign(self.root, tokenizer, rows).render_prompts()
        self.assertTrue(calls and all(kwargs['enable_thinking'] for kwargs in calls))
        tokenizer.apply_chat_template = lambda prompt, **kwargs: [0] * (recipe.CONTEXT_LENGTH - recipe.MAX_RESPONSE_TOKENS + 1)
        self.assertRaises(AssertionError, Campaign(self.root, tokenizer, rows).render_prompts)

    def test_cleanup_deletes_only_named_resources(self):
        seen = []

        def handler(request):
            seen.append((request.url.path, dict(request.url.params)))
            return httpx.Response(200, json={})

        with patch('lifecycle.httpx.Client', return_value=httpx.Client(transport=httpx.MockTransport(handler))), \
                patch('lifecycle.headers', return_value={}):
            cleanup(self.root, {'account': 'acct', 'trainer_id': 'owned-trainer', 'deployment_id': 'owned-rollout'})
        self.assertEqual(seen, [('/v1/accounts/acct/rlorTrainerJobs/owned-trainer', {}),
                                ('/v1/accounts/acct/deployments/owned-rollout', {'ignoreChecks': 'true'})])


if __name__ == '__main__':
    unittest.main()
