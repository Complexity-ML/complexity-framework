"""Pure checkpoint selection, independent of training and Hub I/O."""
import re

NAME = re.compile(r'^step_[0-9]{7}$')

def describe(name, marker):
    assert NAME.fullmatch(name) and int(name[5:])==marker['step']
    step=marker['step']
    # Migration: startup checkpoints are regular; prior off-cadence saves are interruptions.
    regular=marker.get('regular', step in (2,10,534057) or step%1000==0)
    kind=marker.get('kind', 'regular' if regular else 'interrupt')
    assert kind in ('regular','interrupt')
    return dict(step=step,regular=regular or kind=='regular',kind=kind)

def select_keep(catalog, policy):
    ordered=sorted(catalog,key=lambda name:catalog[name]['step'])
    regular=[name for name in ordered if catalog[name]['regular']]
    interrupted=[name for name in ordered if catalog[name]['kind']=='interrupt']
    keep=set(regular[-policy['regular_checkpoints']:]+interrupted[-policy['interruption_checkpoints']:])
    assert len(keep)<=4
    if ordered: assert ordered[-1] in keep
    return keep
