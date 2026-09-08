"""Read-only validation of this run's locally produced adapter/state files."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from training_protocol_v1.data import epoch_order
from verifier import sha256_file
root=Path('results/lora-pilot-v1')
assert json.loads((root/'completion.json').read_text())['status']=='COMPLETE'
architecture=json.loads(Path('.cache/baseline/model/config.json').read_text())
hidden=architecture['hidden_size']
head=hidden//architecture['num_attention_heads']
qout=head*architecture['num_attention_heads']
vout=head*architecture['num_key_value_heads']
expected={}
for layer in range(24):
 for projection,out in [('q_proj',qout),('v_proj',vout)]:
  prefix=f'model.layers.{layer}.self_attn.{projection}'
  expected[prefix+'.A']=(8,hidden)
  expected[prefix+'.B']=(out,8)
checks=[]
for step in range(6,61,6):
 folder=root/'training'/f'step-{step:04d}'
 adapter=load_file(str(folder/'adapter.safetensors'))
 assert set(adapter)==set(expected)
 assert sum(t.numel() for t in adapter.values())==540672
 assert all(tuple(adapter[n].shape)==shape and adapter[n].dtype==torch.float32 and torch.isfinite(adapter[n]).all() for n,shape in expected.items())
 # The pickle was just written locally by the frozen checkpoint saver; no external file is loaded.
 state=torch.load(folder/'trainer_state.pt',map_location='cpu',weights_only=False)
 assert state['optimizer_step']==step and state['epoch']==step//6-1
 assert state['data_order']==epoch_order(state['epoch'])
 assert state['scheduler']['last_epoch']==step
 assert set(state)>={'python_rng','numpy_rng','torch_rng','optimizer','scheduler','fingerprints'}
 optim=state['optimizer']; group=optim['param_groups'][0]
 assert len(optim['param_groups'])==1 and len(group['params'])==96 and len(optim['state'])==96
 assert group['lr']==0.0002 and group['betas']==(0.9,0.999) and group['eps']==1e-8 and group['weight_decay']==0.0
 for parameter_id,name in zip(group['params'],expected):
  values=optim['state'][parameter_id]
  assert values['step'].item()==step
  for key in ('exp_avg','exp_avg_sq'):
   assert tuple(values[key].shape)==expected[name] and torch.isfinite(values[key]).all()
 metadata=json.loads((folder/'checkpoint.json').read_text())
 assert metadata['step']==step and metadata['adapter_parameters_only'] and not metadata['base_weights_saved']
 for name,expected_hash in state['fingerprints'].items():
  assert sha256_file(Path(name))==expected_hash
 checks.append({'step':step,'status':'PASS','adapter_tensors':len(adapter),'adapter_parameters':540672,'optimizer_parameters_with_state':96,'epoch_order_correct':True,'optimizer_steps_correct':True,'rng_states_present':True,'base_weights_saved':False,'adapter_sha256':sha256_file(folder/'adapter.safetensors'),'trainer_state_sha256':sha256_file(folder/'trainer_state.pt')})
 assert all(folder.joinpath(name).is_file() for name in ('adapter.safetensors','trainer_state.pt','checkpoint.json'))
result={'status':'PASS','checkpoints_checked':10,'checks':checks,'model_loads':0,'model_generations':0,'optimizer_updates':0,'holdout_content_reads':0}
output=root/'checkpoint_validation.json'
if output.exists():
 assert json.loads(output.read_text())==result, 'Saved checkpoint validation differs'
else:
 with output.open('x') as f:
  json.dump(result,f,indent=2); f.write('\n')
print('PASS: all ten adapter/state checkpoints have exact LoRA tensors, finite weights/moments, correct steps, epoch orders and saved RNG state.')
