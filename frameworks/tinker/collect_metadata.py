"""Retain remote run/checkpoint metadata and a billing snapshot, without weights."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    root=args.root.resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=root/f'service-metadata-{stamp}'
    out.mkdir()
    cli=str(Path(sys.executable).parent/'tinker')
    jobs=[]
    for path in sorted(root.glob('*/model-info.json')):
        model_id=json.loads(path.read_text())['model_id']
        for kind,command in [('run',['run','info',model_id]),('checkpoints',['checkpoint','list','--run-id',model_id])]:
            jobs.append((f'{path.parent.name}-{kind}',[cli,'-f','json',*command]))
    times=[datetime.fromisoformat(json.loads(line)['utc'])
           for path in root.glob('*/events.jsonl') for line in path.read_text().splitlines()]
    start=min(times).replace(minute=0,second=0,microsecond=0).isoformat().replace('+00:00','Z')
    end=(max(times).replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)).isoformat().replace('+00:00','Z')
    jobs.append(('billing',[cli,'-f','json','billing','usage',start,end]))
    def run(job):
        name,command=job
        with (out/f'{name}.json').open('w') as stdout, (out/f'{name}.error.log').open('w') as stderr:
            result=subprocess.run(command,stdout=stdout,stderr=stderr)
        return {'name':name,'command':command,'exit_code':result.returncode}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(run,jobs))
    (out/'commands.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'directory':str(out),'results':[{k:r[k] for k in ('name','exit_code')} for r in results]},indent=2))


if __name__=='__main__':
    main()
