"""Execute the real updater orchestration with stubbed transport and engines."""

import ast
from pathlib import Path
from types import SimpleNamespace

path=Path(__file__).resolve().parent/'source/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py'
tree=ast.parse(path.read_text())
function=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='update_weights')
function.decorator_list=[]
module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
for rank in (0,1):
    events=[]
    def remote(name):
        return SimpleNamespace(remote=lambda: events.append(name))
    engine=SimpleNamespace(**{name:remote(name) for name in
        ('pause_generation','flush_cache','begin_weight_update','end_weight_update','continue_generation')})
    namespace={'dist':SimpleNamespace(get_rank=lambda:rank,barrier=lambda **kwargs:events.append('barrier')),
               'ray':SimpleNamespace(get=lambda refs:None),'get_gloo_group':lambda:None,'tqdm':lambda **kwargs:None}
    exec(compile(module,str(path),'exec'),namespace)
    updater=SimpleNamespace(weight_version=0,rollout_engines=[engine],quantization_config=None,
        _group_name='test',_is_pp_src_rank=rank==0,_send_weights=lambda progress:events.append('transfer'))
    namespace['update_weights'](updater)
    assert updater.weight_version==1
    expected=['pause_generation','flush_cache','begin_weight_update','barrier','transfer','end_weight_update','continue_generation','barrier'] if rank==0 else ['barrier','transfer','barrier']
    assert events==expected,(rank,events)
print('Native updater opens and closes one session per broadcast; both rank contracts passed.')
