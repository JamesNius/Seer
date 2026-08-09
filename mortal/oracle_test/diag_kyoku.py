import gzip, json

LOG_PATH =r'E:\天凤训练数据\2021\01\2021010100gm-00a9-0000-0af99d68.mjson'

raw = gzip.open(LOG_PATH, 'rt', encoding='utf-8').read()
idx = -1
game = 0
for line in raw.splitlines():
    ev = json.loads(line)
    t = ev.get('type')
    if t == 'start_game':
        game += 1
        print(f'=== 第{game}个游戏: {ev.get("names")}')
    elif t == 'start_kyoku':
        idx += 1
        print(f'  at_kyoku={idx}: {ev["bakaze"]}{ev["kyoku"]} 本场{ev.get("honba", 0)}')