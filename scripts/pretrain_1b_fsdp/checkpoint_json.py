"""Explicit JSON codec for DCP metadata; no imports or callables from input data."""
import dataclasses
import json
import math
import os
from pathlib import Path, PurePosixPath
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    Metadata, TensorStorageMetadata, BytesStorageMetadata, ChunkStorageMetadata,
    TensorProperties, MetadataIndex, StorageMeta,
)
from data import sha

CLASSES={c.__name__:c for c in (Metadata,TensorStorageMetadata,BytesStorageMetadata,
    ChunkStorageMetadata,TensorProperties,MetadataIndex,StorageMeta,_StorageInfo)}
VALUES={str(v):v for v in (torch.float16,torch.bfloat16,torch.float32,torch.float64,
    torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64,torch.bool,
    torch.complex64,torch.complex128,torch.strided,torch.contiguous_format,
    torch.channels_last,torch.channels_last_3d,torch.preserve_format)}

def encode(value):
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('Nonfinite metadata')
        return value
    if isinstance(value,os.PathLike):return os.fspath(value)
    if type(value) in (torch.dtype,torch.layout,torch.memory_format):
        name=str(value)
        if name not in VALUES:raise ValueError(f'Unsupported torch value: {name}')
        return {'type':'torch','value':name}
    if isinstance(value,torch.Size):return {'type':'size','items':list(value)}
    if dataclasses.is_dataclass(value) and type(value).__name__ in CLASSES:
        return {'type':type(value).__name__,'fields':{f.name:encode(getattr(value,f.name)) for f in dataclasses.fields(value)}}
    if type(value) is dict:return {'type':'dict','items':[[encode(k),encode(v)] for k,v in value.items()]}
    if type(value) in (list,tuple):return {'type':type(value).__name__,'items':[encode(v) for v in value]}
    raise TypeError(f'Unsupported metadata type: {type(value)}')

def decode(value,depth=0):
    if depth>100:raise ValueError('Metadata nesting limit')
    if value is None or type(value) in (str,int,float,bool):return value
    if type(value) is not dict:raise ValueError('Invalid tagged metadata')
    tag=value.get('type')
    if tag=='torch':return VALUES[value['value']]
    if tag=='size':
        items=value['items']
        if not all(type(v) is int and v>=0 for v in items):raise ValueError('Invalid tensor dimensions')
        return torch.Size(items)
    if tag in ('list','tuple','dict'):
        items=value['items']
        if tag=='dict':return {decode(k,depth+1):decode(v,depth+1) for k,v in items}
        items=[decode(v,depth+1) for v in items]
        return items if tag=='list' else tuple(items)
    if tag not in CLASSES:raise ValueError(f'Unsupported metadata tag: {tag}')
    cls=CLASSES[tag]; fields=value['fields']
    if set(fields)!={f.name for f in dataclasses.fields(cls)}:raise ValueError(f'Unexpected fields for {tag}')
    return cls(**{k:decode(v,depth+1) for k,v in fields.items()})

def write_metadata(metadata,path):
    path=Path(path); tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'format':'tr-hash-dcp-json-v1','metadata':encode(metadata)},allow_nan=False,separators=(',',':')))
    tmp.replace(path)

def read_metadata(path):
    path=Path(path)
    if path.stat().st_size>64*2**20:raise ValueError('Metadata file too large')
    value=json.loads(path.read_text())
    if value.get('format')!='tr-hash-dcp-json-v1':raise ValueError('Unsupported JSON checkpoint format')
    metadata=decode(value['metadata'])
    if type(metadata) is not Metadata:raise ValueError('Expected DCP Metadata')
    for index,item in metadata.storage_data.items():
        if type(index) is not MetadataIndex or type(item) is not _StorageInfo:raise ValueError('Invalid storage mapping')
        p=PurePosixPath(item.relative_path)
        if p.is_absolute() or len(p.parts)!=1 or not p.name.endswith('.distcp'):raise ValueError('Invalid shard path')
        if type(item.offset) is not int or type(item.length) is not int or min(item.offset,item.length)<0:raise ValueError('Invalid shard range')
        if item.transform_descriptors:raise ValueError('Unsupported storage transforms')
    return metadata

class JsonFileSystemReader(FileSystemReader):
    def read_metadata(self,*args,**kwargs):
        metadata=read_metadata(Path(self.path)/'metadata.json')
        if metadata.storage_meta is None:metadata.storage_meta=StorageMeta()
        metadata.storage_meta.load_id=self.load_id
        return metadata

def checkpoint_reader(directory):
    directory=Path(directory)
    return JsonFileSystemReader(directory) if (directory/'metadata.json').exists() else FileSystemReader(directory)

def prepare_upload(source, staging_root):
    """Convert only a trusted local checkpoint produced by this trainer.

    Large files are hard-linked. Originals stay untouched for local resumption.
    The original complete manifest must match before the local pickle is read.
    """
    source=Path(source); staging=Path(staging_root)/source.name
    original=json.loads((source/'complete.json').read_text()); source_digest=sha(source/'complete.json')
    if (staging/'complete.json').exists():
        cached=json.loads((staging/'complete.json').read_text())
        if cached.get('source_manifest_sha256')==source_digest:return staging
        raise ValueError('Existing staging snapshot has different source')
    import shutil
    if staging.exists():shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name,expected in original['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid manifest path')
        src=source/name
        if src.stat().st_size!=expected['bytes'] or sha(src)!=expected['sha256']:raise ValueError(f'Local checksum mismatch: {name}')
        if name=='distributed/.metadata':continue
        dest=staging/name;dest.parent.mkdir(parents=True,exist_ok=True);os.link(src,dest)
    metadata=FileSystemReader(source/'distributed').read_metadata()
    write_metadata(metadata,staging/'distributed/metadata.json')
    # Equality of the normalized representation includes planner and storage maps.
    if encode(metadata)!=encode(read_metadata(staging/'distributed/metadata.json')):raise ValueError('JSON metadata roundtrip mismatch')
    exported=dict(original)
    exported['files']={k:v for k,v in original['files'].items() if k!='distributed/.metadata'}
    file=staging/'distributed/metadata.json'
    exported['files']['distributed/metadata.json']={'bytes':file.stat().st_size,'sha256':sha(file)}
    exported['source_manifest_sha256']=source_digest;exported['metadata_format']='tr-hash-dcp-json-v1'
    (staging/'complete.json').write_text(json.dumps(exported,indent=2))
    return staging
