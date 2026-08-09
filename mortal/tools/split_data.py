"""
将年份目录下的 .mjson 文件按每 10000 个分入子目录
用法: python split_data.py E:/天凤训练数据/2021
产出: 2021/01/ 2021/02/ 2021/03/ ...
"""
import os
import sys
import shutil
import glob

def split_year(year_dir: str, batch_size: int = 10000):
    files = sorted(glob.glob(os.path.join(year_dir, '*.mjson')))
    if not files:
        # try subdirs
        files = sorted(glob.glob(os.path.join(year_dir, '**', '*.mjson'), recursive=True))
    total = len(files)
    print(f'{year_dir}: {total} files')

    n_groups = (total + batch_size - 1) // batch_size
    for g in range(n_groups):
        sub = os.path.join(year_dir, f'{g+1:02d}')
        os.makedirs(sub, exist_ok=True)
        chunk = files[g * batch_size : (g + 1) * batch_size]
        for f in chunk:
            dst = os.path.join(sub, os.path.basename(f))
            if os.path.abspath(f) != os.path.abspath(dst):
                shutil.move(f, dst)
        print(f'  {sub}: {len(chunk)} files')

    # remove empty subdirs (if any)
    for entry in os.listdir(year_dir):
        path = os.path.join(year_dir, entry)
        if os.path.isdir(path) and entry not in [f'{i+1:02d}' for i in range(n_groups)]:
            try:
                if not os.listdir(path):
                    os.rmdir(path)
            except OSError:
                pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python split_data.py <year_dir> [batch_size]')
        sys.exit(1)
    year_dir = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    split_year(year_dir, batch_size)
