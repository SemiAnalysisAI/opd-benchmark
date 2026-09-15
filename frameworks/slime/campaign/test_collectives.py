"""Exercise real CUDA collectives on the process groups selected for this run."""

from datetime import timedelta
import os
import torch
import torch.distributed as dist

# Megatron can retain these aliases before Slime installs its module wrappers.
from torch.distributed import all_gather_into_tensor, reduce_scatter_tensor, all_to_all_single
from slime.utils.reloadable_process_group import monkey_patch_torch_dist, register_default_process_group, ReloadableProcessGroup

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(seconds=60))
native_groups=os.environ.get('SLIME_NATIVE_PROCESS_GROUPS')=='1'
if not native_groups:
    register_default_process_group(timedelta(seconds=60))
    monkey_patch_torch_dist()
group=dist.new_group([0,1],backend='nccl')
assert isinstance(group,ReloadableProcessGroup) is not native_groups
value=torch.tensor([float(rank)],device='cuda')
gathered=torch.empty(2,device='cuda')
all_gather_into_tensor(gathered,value,group=group)
assert gathered.tolist()==[0.,1.]
reduced=torch.empty(1,device='cuda')
reduce_scatter_tensor(reduced,torch.full((2,),float(rank+1),device='cuda'),group=group)
assert reduced.item()==3.
exchanged=torch.empty(2,device='cuda')
all_to_all_single(exchanged,torch.tensor([rank*10.,rank*10.+1],device='cuda'),group=group)
assert exchanged.tolist()==[float(rank),float(rank+10)]
coalesced=torch.empty(1,device='cuda')
with dist._coalescing_manager(group=group,async_ops=True) as manager:
    reduce_scatter_tensor(coalesced,torch.full((2,),float(rank+1),device='cuda'),group=group,async_op=True)
manager.wait()
assert coalesced.item()==3.
asynchronous=torch.empty(2,device='cuda')
work=all_gather_into_tensor(asynchronous,value,group=group,async_op=True)
work.wait()
assert asynchronous.tolist()==[0.,1.]
dist.barrier(group=group)
if rank==0:
    print(f'CUDA all-gather, reduce-scatter, all-to-all and asynchronous/coalesced waits passed; native_groups={native_groups}.')
dist.destroy_process_group()
