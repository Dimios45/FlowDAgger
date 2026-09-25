"""Pinned two-dataset offline training; never deploys or commands hardware."""
import argparse
import json
from pathlib import Path
from .autonomous_dataset import prepare
from .data import write_json
from .snapshots import fetch, matches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='JSON training defaults; explicit flags override them')
    p.add_argument('--root',default='outputs/yam-flowdagger-v2')
    p.add_argument('--human-root',default='outputs/yam-flowdagger-v1')
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--weight-decay',type=float,default=1e-3)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--timing-mode',choices=('restore-native','recorded'),default='restore-native')
    p.add_argument('--checkpoint-selection',choices=('best','last'),default='best')
    p.add_argument('--all-training',action='store_true',help='Use all eligible episodes; no validation or test evaluation')
    p.add_argument('--prepare-only',action='store_true')
    preliminary,_=p.parse_known_args()
    if preliminary.config:
        config=json.loads(preliminary.config.read_text())
        allowed={'epochs','patience','learning_rate','weight_decay','batch_size','seed','timing_mode','checkpoint_selection','all_training'}
        if set(config)-allowed:
            p.error(f'Unknown config fields: {set(config)-allowed}')
        p.set_defaults(**config)
    a=p.parse_args()
    if a.timing_mode not in ('restore-native','recorded'):
        p.error('Invalid timing_mode')
    if a.all_training and a.checkpoint_selection != 'last':
        p.error('--all-training requires --checkpoint-selection last')
    root,human,out=Path(a.root),Path(a.human_root),Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'resolved-config.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    if (out/'actor.pt').exists():
        raise ValueError('Use a fresh run directory; an existing actor is not overwritten')
    def status(stage,**extra):
        write_json(out/'status.json',dict(stage=stage,**extra))
        print(stage,extra,flush=True)
    try:
        status('verifying-human-snapshot')
        snapshots=Path(__file__).with_name('snapshots')
        source=json.loads((human/'manifest.json').read_text())
        human_snapshot=json.loads((snapshots/'pi05-pink-towel-dagger6-files.json').read_text())
        for entry in human_snapshot['files']:
            # The existing cache derives from these exact data, metadata and video files.
            if entry['path'].startswith(('data/','meta/','videos/')):
                if not matches(Path(source['dataset'])/entry['path'],entry):
                    raise ValueError(f"Existing HITL cache source differs from pinned upload: {entry['path']}")
        auto_name='pi05-molmoact2-yam-autonmous-buffer'
        snapshot=snapshots/(auto_name+'-files.json')
        dataset=root/'datasets'/auto_name
        status('downloading-autonomous-snapshot')
        fetch(snapshot,dataset)
        status('preparing-timing-filtered-preservation')
        preparation_root=root/'recorded-cadence' if a.timing_mode == 'recorded' else root
        if a.all_training:
            preparation_root=preparation_root/'all-training'
        preparation_root.mkdir(parents=True,exist_ok=True)
        manifest=preparation_root/'autonomous-manifest.json'
        summary=prepare(dataset,human/'manifest.json',snapshot,manifest,
                        human/'preservation-review/review.json',seed=a.seed,timing_mode=a.timing_mode,
                        all_training=a.all_training)
        write_json(out/'preparation.json',summary)
        if a.prepare_only:
            status('prepared')
            return
        from .backend import NativeBackend
        from .pipeline import invert_dataset,read_cache,train,evaluate
        from .core import load_actor
        from .preservation import load_preservation_cache
        from .verify import verify
        status('loading-frozen-base')
        backend=NativeBackend('/home/sra/molmoact2/outputs/models/molmoact2-yam-pi05',
                              '/home/sra/molmoact2/outputs/models/paligemma-tokenizer','cuda')
        cache=preparation_root/'autonomous-cache'
        status('inverting-autonomous-targets')
        invert_dataset(backend,manifest,cache)
        status('verifying-native-parity-and-inversion')
        checks=dict(human=verify(backend,human/'cache'),autonomous=verify(backend,cache))
        write_json(out/'verification.json',checks)
        hm,he=read_cache(human/'cache',accepted_only=False)
        am,ae,at=load_preservation_cache(cache,hm)
        write_json(out/'buffer-counts.json',dict(accepted_autonomous_train=len(at),
            autonomous_candidates=len(ae),human_candidates=len(he)))
        status('training')
        train(backend,human/'cache',out/'actor.pt',epochs=a.epochs,patience=a.patience,
              batch_size=a.batch_size,learning_rate=a.learning_rate,weight_decay=a.weight_decay,
              seed=a.seed,preservation_cache=cache,checkpoint_selection=a.checkpoint_selection,
              all_training=a.all_training)
        if a.all_training:
            status('complete',actor=str(out/'actor.pt'),checkpoint_selection='last',
                   validation_evaluated=False,test_evaluated=False,robot_commands_issued=False)
            return
        status('comparing-held-out-validation')
        actor,_=load_actor(out/'actor.pt',hm['identity'],'cuda')
        reports={}
        for name,meta,examples in [('human',hm,he),('reconstructed_autonomous',am,ae)]:
            report=evaluate(backend,actor,meta,examples,'validation')
            write_json(out/f'validation-{name}.json',report)
            reports[name]=report['mean_normalized_rmse']
        reports['equal_buffer_mean']={name:sum(r[name] for r in reports.values())/2
                                      for name in ('steered','zero','random')}
        write_json(out/'comparison.json',reports)
        status('complete',actor=str(out/'actor.pt'),comparison=reports,
               test_evaluated=False,robot_commands_issued=False)
    except Exception as e:
        status('failed',error=f'{type(e).__name__}: {e}')
        raise


if __name__=='__main__':
    main()
