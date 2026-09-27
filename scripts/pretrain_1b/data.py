"""Pinned, non-replaying 16K corpus reader with a bounded, verified Hub cache."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from complexity.training.corpus_mixture import _HubShardCache

ROOT = Path(os.environ.get("TR_HASH_RUN_DIR", "artifacts/pretrain-1b-100b")).resolve()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

class StableCache(_HubShardCache):
    def get(self, filename, *, expected_bytes=None, expected_sha256=None):
        # Preserve mtime: the framework verification marker uses it to avoid
        # hashing the full 2 GB shard again for every 16K training sequence.
        relative=Path(filename)
        assert not relative.is_absolute() and '..' not in relative.parts
        destination=self.root/relative
        lock=self._lock(self.root/'.locks'/'downloads'/(hashlib.sha256(filename.encode()).hexdigest()+'.lock'))
        with lock:
            if not destination.is_file(): destination=self._download(filename)
            try:
                self._validate(destination,expected_bytes=expected_bytes,expected_sha256=expected_sha256)
            except ValueError:
                destination.unlink(missing_ok=True)
                self._verification_marker(destination).unlink(missing_ok=True)
                destination=self._download(filename)
                self._validate(destination,expected_bytes=expected_bytes,expected_sha256=expected_sha256)
            stat=destination.stat()
            os.utime(destination,ns=(time.time_ns(),stat.st_mtime_ns))
        self._evict(exclude={destination})
        return destination

class Corpus:
    def __init__(self, rank=0):
        self.plan = json.loads((ROOT/'data-plan.json').read_text())
        self.recipe = json.loads((ROOT/'recipe.json').read_text())
        for filename, expected in [('data-plan.json', self.recipe['data_plan_sha256']),
                                   ('model.json', self.recipe['model_sha256']),
                                   ('schedule.npy', self.plan['schedule_sha256']),
                                   ('source-update-offsets.npy', self.plan['offsets_sha256'])]:
            assert sha(ROOT/filename) == expected, filename
        self.schedule = np.load(ROOT/'schedule.npy', mmap_mode='r')
        self.offsets = np.load(ROOT/'source-update-offsets.npy', mmap_mode='r')
        self.length = self.recipe['context']
        self.ends = []
        for source in self.plan['sources']:
            capacities = [(s['tokens']-1)//self.length*self.length for s in source['manifest']['shards']]
            self.ends.append(np.cumsum(capacities))
            assert source['train_start_token'] + source['updates']*self.plan['tokens_per_update'] <= sum(capacities)
        self.cache = StableCache(repo_id=self.plan['dataset'], cache_dir=ROOT/'cache',
            revision=self.plan['revision'], token=(ROOT/'runtime/hf-token').read_text().strip() if (ROOT/'runtime/hf-token').exists() else None, max_cache_bytes=self.recipe['cache_gib']*2**30,
            prefetch_shards=4 if rank == 0 else 0)

    def location(self, source_id, offset):
        ends = self.ends[source_id]
        index = int(np.searchsorted(ends, offset, side='right'))
        start = int(ends[index-1]) if index else 0
        source = self.plan['sources'][source_id]
        shard = source['manifest']['shards'][index]
        return 'corpora/'+source['name']+'/'+shard['file'], shard, offset-start

    def sequence(self, source_id, offset):
        filename, shard, local = self.location(source_id, offset)
        assert local+self.length+1 <= shard['tokens']
        with self.cache.pinned(filename, expected_bytes=shard['bytes'], expected_sha256=shard['sha256']) as path:
            with open(path, 'rb') as f:
                f.seek(local*2)
                a = np.frombuffer(f.read((self.length+1)*2), dtype='<u2').astype(np.int64)
        assert len(a)==self.length+1 and a.max()<32000
        return a

    def batch(self, update, rank, validation_source=None, microstep=0):
        source_id = int(self.schedule[update]) if validation_source is None else validation_source
        source = self.plan['sources'][source_id]
        offset = source['train_start_token']+int(self.offsets[update])*self.plan['tokens_per_update'] if validation_source is None else 0
        offset += (microstep*self.recipe['world_size']+rank)*self.recipe['microbatch']*self.length
        return np.stack([self.sequence(source_id, offset+j*self.length) for j in range(self.recipe['microbatch'])])

    def prefetch(self, update):
        for step in range(update, min(update+1536, len(self.schedule))):
            sid = int(self.schedule[step]); source = self.plan['sources'][sid]
            offset = source['train_start_token']+int(self.offsets[step])*self.plan['tokens_per_update']
            for off in (offset, offset+self.plan['tokens_per_update']-self.length):
                filename, shard, _ = self.location(sid, off)
                self.cache.prefetch(filename, expected_bytes=shard['bytes'], expected_sha256=shard['sha256'])

if __name__ == '__main__':
    c=Corpus(); c.prefetch(0)
    for sid in range(len(c.plan['sources'])):
        c.sequence(sid, 0)
        print(json.dumps({'event':'source_ready','source':c.plan['sources'][sid]['name']}), flush=True)
