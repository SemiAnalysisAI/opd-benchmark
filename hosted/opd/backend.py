"""The service behind sft.py, opd.py and mopd.py.

The scripts use only the Tinker client API: a training client (forward_backward, optim_step,
save_state, save_weights_for_sampler) and sampling clients (sample, compute_logprobs).
A backend connects and hands out those clients:

- Tinker: the hosted Tinker service.
- Fireworks: the dedicated Training API (Tinker-compatible), attached to a lease that
  fireworks_lease.py provisions (a trainer and a sampling deployment, under a budget watchdog).
- FireworksServerless: the per-token serverless Training API for the student; uploaded teachers
  run on their own deployments from a teacher-only lease (fireworks_lease.py up --teachers-only).
"""
import json
import os
from pathlib import Path
import sys

import tinker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.credentials import api_key, local_config  # noqa: E402


class Tinker:
    """Tinker's hosted service. The key comes from $TINKER_API_KEY, or from the `~/.zprofile` label in
    config/tinker.local.json; if that file sets expected_email and expected_org, other accounts are refused."""
    name = 'tinker'

    def __init__(self, experiment):
        settings = local_config('tinker')
        os.environ['TINKER_API_KEY'] = api_key('TINKER_API_KEY', settings.get('api_key_profile_label'))
        self.account = self.verify(settings)
        self.service = tinker.ServiceClient(user_metadata={'experiment': experiment})

    @staticmethod
    def verify(settings):
        import httpx
        from tinker.cli.auth_api import TinkerAuthApi
        with httpx.Client(timeout=30) as client:
            identity = TinkerAuthApi(client).get_self_api_key(os.environ['TINKER_API_KEY'])
        account = {'email': identity.details.user_details.email, 'organization': identity.details.org_details.name}
        expected = settings.get('expected_email'), settings.get('expected_org')
        if any(expected) and (account['email'], account['organization']) != expected:
            raise RuntimeError(f'Account guard rejected {account["email"]} / {account["organization"]}')
        return {**account, 'account_guard_passed': any(expected)}

    def capabilities(self, model):
        caps = self.service.get_server_capabilities()
        assert model in [m.model_name for m in caps.supported_models], f'{model} is not served'
        return caps.model_dump()

    def new_student(self, model, rank, seed, metadata):
        """A fresh LoRA on the base model."""
        return self.service.create_lora_training_client(model, rank=rank, seed=seed, user_metadata=metadata)

    def resume_student(self, state_path):
        """A student restored from an optimizer-state checkpoint."""
        return self.service.create_training_client_from_state_with_optimizer(state_path)

    def sampler(self, model=None, path=None):
        """A sampling client for the base model or for saved sampler weights (a frozen teacher)."""
        return self.service.create_sampling_client(**({'model_path': path} if path else {'base_model': model}))

    def sync(self, train, name):
        """Publish the student's current weights and return a sampling client for them, plus their path."""
        path = train.save_weights_for_sampler(name=name).result().path
        return self.sampler(path=path), path

    def use_tokenizers(self, student, teacher):
        pass

    def close(self):
        pass


class Fireworks:
    """Fireworks' dedicated Training API on the lease in $FIREWORKS_LEASE (see fireworks_lease.py).

    Differences from Tinker that the scripts do not see:
    - One LoRA session per trainer, so runs on one lease go one after another.
    - Sampling hot-loads a saved snapshot into the lease's single deployment, so one process
      samples from one set of weights at a time.
    - A base-model sampler is the snapshot of a fresh LoRA (its B matrices are zero).
    - With --policy-lag 1 the next batch is in flight during the hot load. The deployment's SYNC transition lets
      in-flight requests finish on the old weights, but requests still queued client-side (beyond the concurrency
      window) are sent after it and sample the new weights, so their true lag is 0, not the recorded 1. The loss
      stays correct: it uses the logprobs the deployment returned.
    - A frozen teacher is an uploaded model on its own deployment: teachers.json holds
      `fireworks-deployment://accounts/<account>/deployments/<id>` paths (fireworks_lease.py --teacher).
    """
    TEACHER_PREFIX = 'fireworks-deployment://'
    INFERENCE_URL = 'https://api.fireworks.ai'
    # Requests in flight per sampler, per deployment replica. The SDK's adaptive window grows while the deployment
    # keeps up and halves on overload (HTTP 429), but its default start of 8 barely grows during multi-minute
    # thinking responses; a fixed 256 on one 2-GPU replica was refused with 429s. Start at 32 per replica instead.
    WINDOW_PER_REPLICA = 32
    name = 'fireworks'
    MODELS = {'Qwen/Qwen3.6-35B-A3B': 'accounts/fireworks/models/qwen3p6-35b-a3b'}
    SHAPE = 'accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k-lora/versions/a1k9h54n'
    MAX_LORA_RANK = 64
    CONTEXT_LENGTH = 32_768

    def __init__(self, experiment):
        from fireworks.training.sdk import FiretitanServiceClient
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'fireworks'))
        from lifecycle import check, verify_account  # The budget check and account check of hosted/fireworks.
        self.check = check
        self.lease = Path(os.environ['FIREWORKS_LEASE']).resolve()
        self.resources = json.loads((self.lease/'resources.json').read_text())
        self.check(self.lease)
        settings = local_config('fireworks')
        os.environ['FIREWORKS_API_KEY'] = api_key('FIREWORKS_API_KEY', settings.get('api_key_profile_label'))
        self.account = {**verify_account(self.resources['account']), 'lease': str(self.lease),
                        'trainer_id': self.resources['trainer_id'], 'deployment_id': self.resources['deployment_id']}
        self.service = FiretitanServiceClient.from_firetitan_config(**service_config(self.resources, experiment))
        # The deployment's completions API takes text stop sequences; the sampler converts token ids with this.
        from transformers import AutoTokenizer
        from common import MODEL, REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

    def capabilities(self, model):
        assert model in self.MODELS, f'{model} has no Fireworks mapping'
        return {'model': self.MODELS[model], 'training_shape': self.SHAPE, 'max_lora_rank': self.MAX_LORA_RANK,
                'max_context_length': self.CONTEXT_LENGTH, 'resources': self.resources}

    def new_student(self, model, rank, seed, metadata):
        self.check(self.lease)
        assert rank <= self.MAX_LORA_RANK
        return self.service.create_lora_training_client(base_model=self.MODELS[model], rank=rank, seed=seed,
                                                        alpha=32, user_metadata=metadata)

    def resume_student(self, state_path):
        self.check(self.lease)
        return self.service.create_training_client_from_state_with_optimizer(state_path)

    def sampler(self, model=None, path=None):
        """A sampler for saved weights (hot-loaded into the deployment), or for the base model."""
        self.check(self.lease)
        if path and path.startswith(self.TEACHER_PREFIX):
            from fireworks.training.sdk.client import FiretitanSamplingClient
            from fireworks.training.sdk.sampling import DeploymentSampler
            return FiretitanSamplingClient(DeploymentSampler(
                inference_url=self.INFERENCE_URL, model=path[len(self.TEACHER_PREFIX):],
                api_key=os.environ['FIREWORKS_API_KEY'], tokenizer=self.tokenizer,
                concurrency_controller=self.concurrency(replicas=1)))
        if path:
            return self.service.create_sampling_client(model_path=path, tokenizer=self.tokenizer,
                                                       concurrency_controller=self.concurrency())
        client, _ = self.sync(self.new_student(model, self.MAX_LORA_RANK, 0, {'purpose': 'base-model sampler'}), 'base')
        return client

    def sync(self, train, name):
        self.check(self.lease)
        path = train.save_weights_for_sampler(name=name).result().path
        return self.service.create_sampling_client(model_path=path, tokenizer=self.tokenizer, training_client=train,
                                                   concurrency_controller=self.concurrency()), path

    def concurrency(self, replicas=None):
        from fireworks.training.sdk.concurrency import AdaptiveConcurrencyController
        replicas = replicas or self.resources.get('sampler_replicas', 1)
        return AdaptiveConcurrencyController(initial_window=self.WINDOW_PER_REPLICA * replicas,
                                             max_window=4 * self.WINDOW_PER_REPLICA * replicas)

    def use_tokenizers(self, student, teacher):
        pass  # This backend serves the recipe's model and loads its tokenizer itself.

    def close(self):
        self.service.close()


class FireworksServerless:
    """Fireworks' serverless Training API: a pooled, per-token LoRA trainer and sampler for the student.

    Teachers are uploaded models on their own deployments (`fireworks-deployment://...` paths in
    teachers.json), from a teacher-only lease in $FIREWORKS_LEASE whose budget check applies here too.
    Serverless snapshots can only be sampled while this session is alive, and checkpoint names must be
    at most 17 characters.
    """
    name = 'fireworks-serverless'
    URL = 'https://api.fireworks.ai/training/v1/serverless'
    MODELS = {'Qwen/Qwen3.8-27B': 'accounts/fireworks/models/qwen3p8-27b'}
    WINDOW = 64  # Adaptive concurrency start; it halves on 429s.

    def __init__(self, experiment):
        from fireworks.training.sdk import FiretitanServiceClient
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'fireworks'))
        from lifecycle import check, verify_account
        self.check = check
        self.lease = Path(os.environ['FIREWORKS_LEASE']).resolve() if os.environ.get('FIREWORKS_LEASE') else None
        settings = local_config('fireworks')
        os.environ['FIREWORKS_API_KEY'] = api_key('FIREWORKS_API_KEY', settings.get('api_key_profile_label'))
        account = settings.get('account') or os.environ.get('FIREWORKS_ACCOUNT_ID')
        self.account = {**verify_account(account), 'endpoint': self.URL, 'teacher_lease': str(self.lease)}
        self.service = FiretitanServiceClient(api_key=os.environ['FIREWORKS_API_KEY'], base_url=self.URL)
        self.tokenizer = self.teacher_tokenizer = None

    def guard(self):
        if self.lease:
            self.check(self.lease)

    def use_tokenizers(self, student, teacher):
        self.tokenizer, self.teacher_tokenizer = student, teacher  # Samplers turn stop-token ids into text with them.

    def capabilities(self, model):
        assert model in self.MODELS, f'{model} is not mapped to a serverless model'
        return {'model': self.MODELS[model], 'endpoint': self.URL, 'teacher_lease': str(self.lease)}

    def new_student(self, model, rank, seed, metadata):
        self.guard()
        return self.service.create_lora_training_client(base_model=self.MODELS[model], rank=rank, seed=seed,
                                                        user_metadata=metadata)

    def resume_student(self, state_path):
        self.guard()
        return self.service.create_training_client_from_state_with_optimizer(state_path)

    def sampler(self, model=None, path=None):
        self.guard()
        from fireworks.training.sdk.concurrency import AdaptiveConcurrencyController
        if path and path.startswith(Fireworks.TEACHER_PREFIX):
            from fireworks.training.sdk.client import FiretitanSamplingClient
            from fireworks.training.sdk.sampling import DeploymentSampler
            return FiretitanSamplingClient(DeploymentSampler(
                inference_url=Fireworks.INFERENCE_URL, model=path[len(Fireworks.TEACHER_PREFIX):],
                api_key=os.environ['FIREWORKS_API_KEY'], tokenizer=self.teacher_tokenizer,
                concurrency_controller=AdaptiveConcurrencyController(initial_window=32, max_window=128)))
        assert path, 'Serverless samples saved snapshots; save a fresh LoRA for the base model'
        return self.service.create_sampling_client(
            model_path=path, tokenizer=self.tokenizer,
            concurrency_controller=AdaptiveConcurrencyController(initial_window=self.WINDOW, max_window=8 * self.WINDOW))

    def sync(self, train, name):
        self.guard()
        assert len(name) <= 17, 'Serverless checkpoint names are at most 17 characters'
        path = train.save_weights_for_sampler(name=name).result().path
        return self.sampler(path=path), path

    def close(self):
        self.service.close()


class SelfHosted:
    """A self-hosted Tinker-compatible server at $TINKER_BASE_URL (see hosted/selfhosted/README.md).

    The servers do not authenticate, but the SDK requires a key that starts with `tml-`. A teacher path of the form
    `base:<model name>` samples a frozen model that the server serves by name (verl-tinker teachers); any other
    path is a sampler checkpoint on the same server. Subclasses set what each server accepts.
    """
    name = 'self-hosted'
    ACCEPTS_SEED = True  # create_lora_training_client(seed=...)
    ACCEPTS_METADATA = True  # user_metadata on the session and on each training client
    BASE_PREFIX = 'base:'

    def __init__(self, experiment):
        self.url = os.environ['TINKER_BASE_URL']
        os.environ.setdefault('TINKER_API_KEY', 'tml-self-hosted')
        server = json.loads(Path(os.environ['TINKER_SERVER_INFO']).read_text()) if os.environ.get('TINKER_SERVER_INFO') else {}
        self.account = {'endpoint': self.url, 'server': self.name, 'server_info': server}
        metadata = {'user_metadata': {'experiment': experiment}} if self.ACCEPTS_METADATA else {}
        self.service = tinker.ServiceClient(base_url=self.url, api_key=os.environ['TINKER_API_KEY'], **metadata)

    def capabilities(self, model):
        caps = self.service.get_server_capabilities()
        assert model in [m.model_name for m in caps.supported_models], f'{model} is not served at {self.url}'
        return caps.model_dump()

    def new_student(self, model, rank, seed, metadata):
        return self.service.create_lora_training_client(
            model, rank=rank, seed=seed if self.ACCEPTS_SEED else None,
            **({'user_metadata': metadata} if self.ACCEPTS_METADATA else {}))

    def resume_student(self, state_path):
        return self.service.create_training_client_from_state_with_optimizer(state_path)

    def sampler(self, model=None, path=None):
        if path and path.startswith(self.BASE_PREFIX):
            return self.service.create_sampling_client(base_model=path[len(self.BASE_PREFIX):])
        return self.service.create_sampling_client(**({'model_path': path} if path else {'base_model': model}))

    def sync(self, train, name):
        path = train.save_weights_for_sampler(name=name).result().path
        return self.sampler(path=path), path

    def use_tokenizers(self, student, teacher):
        pass

    def close(self):
        pass


class SkyRL(SelfHosted):
    """SkyRL's Tinker API server (`python -m skyrl.tinker.api --backend megatron`). LoRA alpha is a server setting, and
    the seed is ignored. A sampler checkpoint stays servable only while its model is loaded, so the server runs with
    --session-timeout-sec -1 and teachers are trained on the same server instance that serves them."""
    name = 'skyrl'


class Miles(SelfHosted):
    """Miles' Tinker gateway (`serve_tinker.py`). It rejects a LoRA seed and user metadata, and caps the rank at the
    server's --lora-rank."""
    name = 'miles'
    ACCEPTS_SEED = False
    ACCEPTS_METADATA = False


class Verl(SelfHosted):
    """verl-tinker (`python -m verl_tinker.start`). One trainable model per server: teachers are frozen models that the
    server config serves by name (`base:<name>` paths), and every training run needs its own server."""
    name = 'verl'


def service_config(resources, experiment=None):
    """from_firetitan_config arguments for the lease's trainer and deployment; closing never deletes them."""
    return dict(
        api_key=os.environ['FIREWORKS_API_KEY'], base_model=Fireworks.MODELS['Qwen/Qwen3.6-35B-A3B'],
        tokenizer_model='Qwen/Qwen3.6-35B-A3B', max_lora_rank=Fireworks.MAX_LORA_RANK,
        training_shape_id=Fireworks.SHAPE, trainer_job_id=resources['trainer_id'],
        deployment_id=resources['deployment_id'], max_context_length=Fireworks.CONTEXT_LENGTH,
        replica_count=resources.get('sampler_replicas', 1),
        **({'deployment_shape': resources['sampler_shape']} if resources.get('sampler_shape') else {}),
        trainer_replica_count=1, hot_load_transition_type=resources.get('hot_load_transition_type', 'SYNC'),
        disable_speculative_decoding=True,
        cleanup_trainer_on_close=False, cleanup_deployment_on_close=None, trainer_timeout_s=1800,
        trainer_pending_timeout_s=1800, deployment_timeout_s=1800, wait_for_trainer_before_deployment=False,
        **({'user_metadata': {'experiment': experiment}} if experiment else {}))


BACKENDS = {'tinker': Tinker, 'fireworks': Fireworks, 'fireworks-serverless': FireworksServerless,
            'skyrl': SkyRL, 'miles': Miles, 'verl': Verl}
