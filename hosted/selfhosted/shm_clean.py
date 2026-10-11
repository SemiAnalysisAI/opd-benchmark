"""Delete /dev/shm files that no process holds open or maps: segments leaked by killed vLLM, PyTorch and NCCL workers."""
import os
from pathlib import Path

held = set()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        for fd in (proc/'fd').iterdir():
            target = os.readlink(fd)
            if target.startswith('/dev/shm/'):
                held.add(target.split(' (deleted)')[0])
        for line in (proc/'maps').read_text().splitlines():
            if '/dev/shm/' in line:
                held.add(line[line.index('/dev/shm/'):].split(' (deleted)')[0])
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        continue
freed = removed = 0
for path in Path('/dev/shm').iterdir():
    if path.is_file() and str(path) not in held:
        freed += path.stat().st_size
        path.unlink()
        removed += 1
print(f'{os.uname().nodename}: removed {removed} unheld /dev/shm files ({freed / 2**30:.1f} GiB), kept {len(held)} held')
