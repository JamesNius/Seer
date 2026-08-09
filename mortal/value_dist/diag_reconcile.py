"""诊断: 用 deltas 对账验证 hand_tracker 算点正确性。

对每条 hora, 比较:
  - calc_agari_point 算出的荣和名义打点(自摸折算)
  - deltas 反推的名义打点(去掉本场 honba*300/自摸 honba*100*3 与供托)
容差内视为一致, 统计一致率。
"""
import gzip, json, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_dist.hand_tracker import HandTracker, WIND_ID

def norm_from_deltas(e):
    """从 deltas 去掉本场/供托,反推名义打点(与 calc 口径对齐)。

    荣和: 放铳者输额 = 名义 + honba*300  (+ 供托归赢家,不影响输家)
    自摸: 赢家进账 = tsumo_total + honba*300 + kyotaku*1000
          等价: 各输家合计输额 = tsumo_total + honba*300
    """
    actor=e['actor']; target=e['target']; deltas=e['deltas']
    honba=e.get('_honba',0); kyotaku=e.get('_kyotaku',0)
    if actor!=target:
        lose = -deltas[target]
        return lose - honba*300
    else:
        # 输家总输额(不含供托,供托来自立直棒已在前局扣除)
        lose_sum = -sum(d for i,d in enumerate(deltas) if i!=actor)
        return lose_sum - honba*300

def run_file(F, max_kyoku=999):
    tracker=HandTracker()
    last_tsumo_pai=None; last_dahai_pai=None
    total=0; ok=0; mism=[]; skip=0
    with gzip.open(F,'rt',encoding='utf-8') as fh:
        for line in fh:
            e=json.loads(line)
            t=e.get('type')
            # 记录 winning tile 上下文
            if t=='tsumo': last_tsumo_pai=e.get('pai'); last_dahai_pai=None
            elif t=='dahai': last_dahai_pai=e.get('pai')
            tracker.update(e)
            if t=='hora':
                total+=1
                actor=e['actor']; is_ron=actor!=e['target']
                wt = last_dahai_pai if is_ron else last_tsumo_pai
                e['_winning_tile']=wt
                e['_honba']=tracker.ks.honba
                e['_kyotaku']=tracker.ks.kyotaku if hasattr(tracker.ks,'kyotaku') else 0
                try:
                    v=tracker.compute_hora_value(e)
                except Exception as ex:
                    skip+=1; continue
                ref=norm_from_deltas(e)
                if ref is not None and abs(v-ref)<=0:
                    ok+=1
                else:
                    mism.append((v,ref,e.get('actor'),e.get('target'),wt,list(tracker.ks.hands[actor].menzen)))
            elif t in ('ryukyoku','end_kyoku','end_game'):
                pass
    return total,ok,skip,mism

if __name__=='__main__':
    files=sorted(glob.glob('E:/天凤训练数据/2014/*.mjson'))[:60]
    GT=GO=GS=0
    for F in files:
        total,ok,skip,mism=run_file(F)
        GT+=total;GO+=ok;GS+=skip
        print(os.path.basename(F),'hora',total,'ok',ok,'skip',skip,'mism',len(mism))
        for m in mism[:3]:
            print('   MISM calc=%s ref=%s actor=%s target=%s wt=%s hand=%s'%m)
    print('TOTAL hora',GT,'ok',GO,'skip',GS,'match_rate', (GO/GT if GT else 0))
