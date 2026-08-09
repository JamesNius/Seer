import gzip, json

SRC = r'E:\天凤训练数据\2021\01\2021010100gm-00a9-0000-0af99d68.mjson'   # 改成你跑 verify 用的同一个文件
TARGET = ('S', 2)        # 南2局; 东1=('E',1), 南1=('S',1), 南3=('S',3)...

raw = gzip.open(SRC, 'rt', encoding='utf-8').read()
print('dump 本局玩家:', json.loads(raw.splitlines()[0]).get('names'))

# 1) 全量解压存 .txt, 可以自己打开翻
out = SRC + '.txt'
open(out, 'w', encoding='utf-8').write(raw)
print(f'已解压到 {out}，共 {len(raw.splitlines())} 行\n')

# 2) 打印目标局的全部事件
in_target = False
for line in raw.splitlines():
    ev = json.loads(line)
    t = ev.get('type')
    if t == 'start_kyoku':
        in_target = (ev.get('bakaze'), ev.get('kyoku')) == TARGET
        if in_target:
            print(f"=== 南2局  本场{ev.get('honba')} 供托{ev.get('kyotaku')} 宝牌指示牌 {ev.get('dora_marker')}")
            for seat, tehai in enumerate(ev['tehais']):
                print(f'  座位{seat} 起手({len(tehai)}张): {" ".join(tehai)}')
        continue
    if not in_target:
        continue
    if t == 'tsumo':
        print(f"  座位{ev['actor']} 摸 {ev['pai']}")
    elif t == 'dahai':
        print(f"  座位{ev['actor']} 打 {ev['pai']}")
    elif t in ('chi', 'pon', 'daiminkan', 'kakan', 'ankan', 'reach', 'reach_accepted', 'dora', 'hora', 'ryukyoku'):
        print(f'  {t}: {ev}')