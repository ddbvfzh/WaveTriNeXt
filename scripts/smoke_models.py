"""CPU forward checks of all five models, both class counts, and optimizers.

Uses random initialization. Does not train or download any weights.
"""
import gc
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from waste_reliability.portable import MODELS, model_module


def main():
    torch.set_num_threads(2)
    records = []
    for alias in MODELS:
        config = json.loads((ROOT / 'configs' / f'{alias}.json').read_text())
        for dataset, classes in [('trashnet',6), ('realwaste',9)]:
            torch.manual_seed(14)
            module = model_module(alias)
            model = module.build_model(classes).eval()
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert count == config['parameter_counts'][dataset], (alias,dataset,count)
            optimizer = module.optimizer_for(model)
            params = [id(p) for group in optimizer.param_groups for p in group['params']]
            assert len(params) == len(set(params))
            assert set(params) == {id(p) for p in model.parameters() if p.requires_grad}
            with torch.inference_mode():
                logits = model(torch.zeros(1,3,224,224))
                assert logits.shape == (1,classes) and torch.isfinite(logits).all()
                if alias == 'wavetrinext':
                    aux = model(torch.zeros(1,3,224,224), return_aux=True)
                    for key in ['logits_cls','logits_fg','logits_bg']:
                        assert aux[key].shape == (1,classes) and torch.isfinite(aux[key]).all()
            records.append({'model':alias,'dataset':dataset,'parameters':count,'forward':'passed','optimizer':'passed'})
            del model, optimizer
            gc.collect()
    print(json.dumps({'status':'passed','device':'cpu','weights':'random; none downloaded','training_performed':False,'checks':records},indent=2))


if __name__ == '__main__':
    main()
