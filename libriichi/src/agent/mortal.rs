use super::{BatchAgent, InvisibleState};
use crate::consts::{obs_shape, ACTION_SPACE};
use crate::mjai::{Event, EventExt, Metadata};
use crate::state::PlayerState;
use crate::{must_tile, tu8};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use crossbeam::sync::WaitGroup;

/// 初始化全局 rayon 池（只允许一次）。`threads=None` 用默认（=逻辑核数）。
/// 已在别处初始化则静默返回 Ok（保持首个配置）。
static RAYON_INIT: AtomicBool = AtomicBool::new(false);

fn init_rayon_pool(threads: Option<usize>) -> Result<()> {
    if RAYON_INIT.swap(true, Ordering::SeqCst) {
        return Ok(());
    }
    match threads {
        None => {
            log::info!("rayon global pool uses default thread count");
            Ok(())
        }
        Some(n) if n == 0 => Err(anyhow::anyhow!("rayon_threads must be >= 1, got {n}")),
        Some(n) => match rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("encode-{i}"))
            .build_global()
        {
            Ok(()) => {
                log::info!("rayon global pool initialized with {n} threads");
                Ok(())
            }
            Err(e) => {
                // 池已在别处初始化（默认或更早配置）：无法再改，回退现有池。
                log::warn!("rayon pool already initialized elsewhere, ignore requested {n} threads: {e}");
                Ok(())
            }
        },
    }
}
use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use parking_lot::Mutex;
use pyo3::intern;
use pyo3::prelude::*;

pub struct MortalBatchAgent {
    engine: PyObject,
    is_oracle: bool,
    version: u32,
    enable_quick_eval: bool,
    enable_rule_based_agari_guard: bool,
    skip_encode: bool,
    /// Pass 1 录制模式：set_scene 照常异步编码，但 get_reaction 不 evaluate，
    /// 用 state 直接推占位动作（省掉每决策点的 wg.wait 串行等待）。
    record_mode: bool,
    name: String,
    player_ids: Vec<u8>,

    actions: Vec<usize>,
    q_values: Vec<[f32; ACTION_SPACE]>,
    masks_recv: Vec<[bool; ACTION_SPACE]>,
    is_greedy: Vec<bool>,
    last_eval_elapsed: Duration,
    last_batch_size: usize,

    evaluated: bool,
    quick_eval_reactions: Vec<Option<Event>>,
    /// record_mode 下：每条 record 的编码结果槽（决策点顺序），rayon 任务
    /// 按 record_id 写入，flush_encode 按序读取，避免并发 push 乱序。
    /// 每元素 = 该 record 的全部有序样本 (obs, mask)。
    record_slots: Option<Arc<Mutex<Vec<Option<Vec<(Array2<f32>, Array1<bool>)>>>>>>,
    /// 主线程按决策点顺序分配的 record 序号（下一个 record 用）
    next_record: usize,

    /// 假想分支（review 用）：set_scene 时为 reach/chi/pon 各预编码一条
    /// "假设执行该动作后的切牌局面" obs，get_reaction 按首推动作取用。
    /// 元素为 (action_label, sync_fields 内样本下标)。
    /// 用 Arc<Mutex> 共享给 rayon 编码闭包；set_scene/get_reaction
    /// 由 Bot 单线程驱动，闭包结束（wg 释放）后才会被 get_reaction 读取。
    pending_followups: Arc<Mutex<Vec<(usize, usize)>>>,

    wg: WaitGroup,
    sync_fields: Arc<Mutex<SyncFields>>,
}

struct SyncFields {
    states: Vec<Array2<f32>>,
    invisible_states: Vec<Array2<f32>>,
    masks: Vec<Array1<bool>>,
    action_idxs: Vec<usize>,
    kan_action_idxs: Vec<Option<usize>>,
}

impl MortalBatchAgent {
    pub fn new(engine: PyObject, player_ids: &[u8]) -> Result<Self> {
        ensure!(player_ids.iter().all(|&id| matches!(id, 0..=3)));

        let (name, is_oracle, version, enable_quick_eval, enable_rule_based_agari_guard, skip_encode, record_mode, rayon_threads) =
            Python::with_gil(|py| {
                let obj = engine.bind_borrowed(py);
                ensure!(
                    obj.getattr("react_batch")?.is_callable(),
                    "missing method react_batch",
                );

                let name = obj.getattr("name")?.extract()?;
                let is_oracle = obj.getattr("is_oracle")?.extract()?;
                let version = obj.getattr("version")?.extract()?;
                let enable_quick_eval = obj.getattr("enable_quick_eval")?.extract()?;
                let enable_rule_based_agari_guard =
                    obj.getattr("enable_rule_based_agari_guard")?.extract()?;
                let skip_encode = obj.getattr("skip_encode").map(|v| v.extract().unwrap_or(false)).unwrap_or(false);
                let record_mode = obj.getattr("record_mode").map(|v| v.extract().unwrap_or(false)).unwrap_or(false);
                // 编码线程池大小：读 Python 引擎的 rayon_threads 属性，缺省 None(=自动)
                let rayon_threads = obj.getattr("rayon_threads")
                    .map(|v| v.extract::<Option<usize>>().ok().flatten())
                    .ok()
                    .flatten();
                Ok((
                    name,
                    is_oracle,
                    version,
                    enable_quick_eval,
                    enable_rule_based_agari_guard,
                    skip_encode,
                    record_mode,
                    rayon_threads,
                ))
            })?;

        // 全局 rayon 池只能初始化一次；按首个 agent 的 rayon_threads 配置。
        // 返回 Err 仅当线程数非法；重复初始化静默忽略（已固定）。
        init_rayon_pool(rayon_threads).with_context(|| "failed to init rayon pool")?;

        let size = player_ids.len();
        let quick_eval_reactions = if enable_quick_eval {
            vec![None; size]
        } else {
            vec![]
        };
        let sync_fields = Arc::new(Mutex::new(SyncFields {
            states: vec![],
            invisible_states: vec![],
            masks: vec![],
            action_idxs: vec![0; size],
            kan_action_idxs: vec![None; size],
        }));

        Ok(Self {
            engine,
            is_oracle,
            version,
            enable_quick_eval,
            enable_rule_based_agari_guard,
            skip_encode,
            record_mode,
            name,
            player_ids: player_ids.to_vec(),

            actions: vec![],
            q_values: vec![],
            masks_recv: vec![],
            is_greedy: vec![],
            last_eval_elapsed: Duration::ZERO,
            last_batch_size: 0,

            evaluated: false,
            quick_eval_reactions,
            record_slots: if record_mode {
                Some(Arc::new(Mutex::new(vec![])))
            } else {
                None
            },
            next_record: 0,
            pending_followups: Arc::new(Mutex::new(vec![])),

            wg: WaitGroup::new(),
            sync_fields,
        })
    }

    fn evaluate(&mut self) -> Result<()> {
        // Wait for all feature encodings to complete.
        mem::take(&mut self.wg).wait();
        let mut sync_fields = self.sync_fields.lock();

        if sync_fields.states.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        self.last_batch_size = sync_fields.states.len();

        (self.actions, self.q_values, self.masks_recv, self.is_greedy) = Python::with_gil(|py| {
            let states: Vec<_> = sync_fields
                .states
                .drain(..)
                .map(|v| PyArray2::from_owned_array(py, v))
                .collect();
            let masks: Vec<_> = sync_fields
                .masks
                .drain(..)
                .map(|v| PyArray1::from_owned_array(py, v))
                .collect();
            let invisible_states: Option<Vec<_>> = self.is_oracle.then(|| {
                sync_fields
                    .invisible_states
                    .drain(..)
                    .map(|v| PyArray2::from_owned_array(py, v))
                    .collect()
            });

            let args = (states, masks, invisible_states);
            self.engine
                .bind_borrowed(py)
                .call_method1(intern!(py, "react_batch"), args)
                .context("failed to execute `react_batch` on Python engine")?
                .extract()
                .context("failed to extract to Rust type")
        })?;

        self.last_eval_elapsed = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO);

        Ok(())
    }

    /// record_mode 结束（Pass 1 录完）后调用：等所有异步编码任务完成，
    /// 返回按 record 分组的编码结果（替代原 RecordEngine.records 的 obs 收集）。
    /// 返回 Python 列表，每元素 = (obs_list, mask_list, inv_list)，三组 numpy 数组；
    /// 顺序与决策点一致，每条 record 的样本数由列表长度自然体现。
    pub(crate) fn flush_encode(&mut self) -> Result<Py<PyAny>> {
        mem::take(&mut self.wg).wait();

        // record_mode：从按序槽位读取（record_slots[0..n]）
        let slots_arc = self.record_slots.take().expect("flush_encode without record_mode");
        let slots = slots_arc.lock();
        let recs = Python::with_gil(|py| {
            let mut out = Vec::new();
            for slot in slots.iter() {
                let samples = slot.as_ref().expect("record slot not filled by flush time");
                let obs_list: Vec<Py<PyAny>> = samples
                    .iter()
                    .map(|(a, _)| PyArray2::from_owned_array(py, a.clone()).into_any().unbind())
                    .collect();
                let mask_list: Vec<Py<PyAny>> = samples
                    .iter()
                    .map(|(_, m)| PyArray1::from_owned_array(py, m.clone()).into_any().unbind())
                    .collect();
                let record = (obs_list, mask_list, Vec::<Py<PyAny>>::new());
                out.push(record.into_pyobject(py).unwrap().unbind().into_any());
            }
            out.into_pyobject(py).unwrap().unbind().into_any()
        });
        Ok(recs)
    }

    fn gen_meta(&self, state: &PlayerState, action_idx: usize) -> Metadata {
        let q_values = self.q_values[action_idx];
        let masks = self.masks_recv[action_idx];
        let is_greedy = self.is_greedy[action_idx];

        let mut mask_bits = 0;
        let q_values_compact = q_values
            .into_iter()
            .zip(masks)
            .enumerate()
            .filter(|&(_, (_, m))| m)
            .map(|(i, (q, _))| {
                mask_bits |= 0b1 << i;
                q
            })
            .collect();

        Metadata {
            q_values: Some(q_values_compact),
            mask_bits: Some(mask_bits),
            is_greedy: Some(is_greedy),
            shanten: Some(state.shanten()),
            at_furiten: Some(state.at_furiten()),
            ..Default::default()
        }
    }
}

/// 假想分支切牌的 tsumogiri 判定：
/// - reach(37)：手牌含摸牌，切摸牌为 tsumogiri
/// - chi/pon(38..=41)：无摸牌，必为手切（false）
fn followup_tsumogiri(label: usize, state: &PlayerState, pai: crate::tile::Tile) -> bool {
    label == 37 && state.last_self_tsumo().is_some_and(|t| t == pai)
}

/// 收集假想分支（review 用）：reach / chi(low/mid/high) / pon 各构造一条
/// "假设执行该动作后的切牌局面" 状态。返回 (label, 执行后状态)。
/// 只判定可执行性（state.update 是否成功），不做 obs 编码。
/// 编码路径与 skip_encode 路径共用，保证样本数、label 完全一致。
fn collect_followups(state: &PlayerState) -> Vec<(usize, PlayerState)> {
    let mut followups = Vec::new();
    let cans = state.last_cans();
    let actor = state.player_id();
    if cans.can_riichi {
        let mut s = state.clone();
        if s.update(&Event::Reach { actor }).is_ok() {
            followups.push((37usize, s));
        }
    }
    if let Some(pai) = state.last_kawa_tile() {
        let target = cans.target_actor;
        let akas = state.akas_in_hand();
        let try_push = |label: usize, can: bool, s: &PlayerState, ev: Event, dst: &mut Vec<(usize, PlayerState)>| {
            if can {
                let mut sc = s.clone();
                if sc.update(&ev).is_ok() {
                    dst.push((label, sc));
                }
            }
        };
        // chi low: consumed = [pai+1, pai+2]
        let chi_low_ev = {
            let first = pai.next();
            let can_akaize = matches!(pai.as_u8(), tu8!(3m)|tu8!(4m)) && akas[0]
                || matches!(pai.as_u8(), tu8!(3p)|tu8!(4p)) && akas[1]
                || matches!(pai.as_u8(), tu8!(3s)|tu8!(4s)) && akas[2];
            let consumed = if can_akaize { [first.akaize(), first.next().akaize()] } else { [first, first.next()] };
            Event::Chi { actor, target, pai, consumed }
        };
        try_push(38, cans.can_chi_low, &state, chi_low_ev, &mut followups);
        // chi mid: consumed = [pai-1, pai+1]
        let chi_mid_ev = {
            let can_akaize = matches!(pai.as_u8(), tu8!(4m)|tu8!(6m)) && akas[0]
                || matches!(pai.as_u8(), tu8!(4p)|tu8!(6p)) && akas[1]
                || matches!(pai.as_u8(), tu8!(4s)|tu8!(6s)) && akas[2];
            let consumed = if can_akaize { [pai.prev().akaize(), pai.next().akaize()] } else { [pai.prev(), pai.next()] };
            Event::Chi { actor, target, pai, consumed }
        };
        try_push(39, cans.can_chi_mid, &state, chi_mid_ev, &mut followups);
        // chi high: consumed = [pai-2, pai-1]
        let chi_high_ev = {
            let last = pai.prev();
            let can_akaize = matches!(pai.as_u8(), tu8!(6m)|tu8!(7m)) && akas[0]
                || matches!(pai.as_u8(), tu8!(6p)|tu8!(7p)) && akas[1]
                || matches!(pai.as_u8(), tu8!(6s)|tu8!(7s)) && akas[2];
            let consumed = if can_akaize { [last.prev().akaize(), last.akaize()] } else { [last.prev(), last] };
            Event::Chi { actor, target, pai, consumed }
        };
        try_push(40, cans.can_chi_high, &state, chi_high_ev, &mut followups);
        // pon
        let pon_ev = {
            let can_akaize = matches!(pai.as_u8(), tu8!(5m)) && akas[0]
                || matches!(pai.as_u8(), tu8!(5p)) && akas[1]
                || matches!(pai.as_u8(), tu8!(5s)) && akas[2];
            let consumed = if can_akaize { [pai.akaize(), pai.deaka()] } else { [pai.deaka(); 2] };
            Event::Pon { actor, target, pai, consumed }
        };
        try_push(41, cans.can_pon, &state, pon_ev, &mut followups);
    }
    followups
}

/// record_mode 下 Pass 1 的占位动作：从 state.last_cans() 直接推一个合法事件，
/// 不做 Q 选择、不依赖编码结果。Pass 1 的 reaction 被 server 丢弃（状态由
/// 输入事件推进），这里只需返回任意合法动作让流程继续。
/// 优先级：和了 > 立直 > 暗杠/加杠 > 吃/碰 > 出牌 > 流局。
fn placeholder_event(state: &PlayerState) -> Result<Event> {
    let cans = state.last_cans();
    let actor = state.player_id();

    if cans.can_agari() {
        return Ok(Event::Hora {
            actor,
            target: cans.target_actor,
            deltas: None,
            ura_markers: None,
        });
    }
    if cans.can_riichi {
        return Ok(Event::Reach { actor });
    }
    if cans.can_ankan {
        let tile = state.ankan_candidates()[0];
        return Ok(Event::Ankan {
            actor,
            consumed: [tile.akaize(), tile, tile, tile],
        });
    }
    if cans.can_kakan {
        let tile = state.kakan_candidates()[0];
        return Ok(Event::Kakan {
            actor,
            pai: tile,
            consumed: [tile.akaize(), tile.deaka(), tile.deaka()],
        });
    }
    if let Some(pai) = state.last_kawa_tile() {
        let akas = state.akas_in_hand();
        let target = cans.target_actor;
        let try_call = |can: bool, ev: Event| if can { Some(ev) } else { None };
        // chi low: consumed = [pai+1, pai+2]
        if let Some(ev) = try_call(cans.can_chi_low, {
            let first = pai.next();
            let can_aka = matches!(pai.as_u8(), tu8!(3m)|tu8!(4m)) && akas[0]
                || matches!(pai.as_u8(), tu8!(3p)|tu8!(4p)) && akas[1]
                || matches!(pai.as_u8(), tu8!(3s)|tu8!(4s)) && akas[2];
            let consumed = if can_aka { [first.akaize(), first.next().akaize()] } else { [first, first.next()] };
            Event::Chi { actor, target, pai, consumed }
        }) {
            return Ok(ev);
        }
        if let Some(ev) = try_call(cans.can_chi_mid, {
            let consumed = if matches!(pai.as_u8(), tu8!(4m)|tu8!(6m)) && akas[0]
                || matches!(pai.as_u8(), tu8!(4p)|tu8!(6p)) && akas[1]
                || matches!(pai.as_u8(), tu8!(4s)|tu8!(6s)) && akas[2]
            {
                [pai.prev().akaize(), pai.next().akaize()]
            } else {
                [pai.prev(), pai.next()]
            };
            Event::Chi { actor, target, pai, consumed }
        }) {
            return Ok(ev);
        }
        if let Some(ev) = try_call(cans.can_chi_high, {
            let last = pai.prev();
            let consumed = if matches!(pai.as_u8(), tu8!(6m)|tu8!(7m)) && akas[0]
                || matches!(pai.as_u8(), tu8!(6p)|tu8!(7p)) && akas[1]
                || matches!(pai.as_u8(), tu8!(6s)|tu8!(7s)) && akas[2]
            {
                [last.prev().akaize(), last.akaize()]
            } else {
                [last.prev(), last]
            };
            Event::Chi { actor, target, pai, consumed }
        }) {
            return Ok(ev);
        }
        if let Some(ev) = try_call(cans.can_pon, {
            let consumed = if matches!(pai.as_u8(), tu8!(5m)) && akas[0]
                || matches!(pai.as_u8(), tu8!(5p)) && akas[1]
                || matches!(pai.as_u8(), tu8!(5s)) && akas[2]
            {
                [pai.akaize(), pai.deaka()]
            } else {
                [pai.deaka(); 2]
            };
            Event::Pon { actor, target, pai, consumed }
        }) {
            return Ok(ev);
        }
    }
    if cans.can_discard {
        // 任意合法出牌：优先 tsumogiri，否则第一候选
        if let Some(pai) = state.last_self_tsumo() {
            return Ok(Event::Dahai {
                actor,
                pai,
                tsumogiri: true,
            });
        }
        if let Some(tile_id) = state
            .discard_candidates_aka()
            .iter()
            .position(|&flag| flag)
        {
            let pai = must_tile!(tile_id);
            return Ok(Event::Dahai {
                actor,
                pai,
                tsumogiri: false,
            });
        }
    }
    if cans.can_ryukyoku {
        return Ok(Event::Ryukyoku { deltas: None });
    }
    anyhow::bail!("placeholder_event: no legal action for {}", state.brief_info());
}

impl MortalBatchAgent {
    /// 返回当前待取用的假想分支样本数（不清空）。
    /// Python 侧按此数从分组结果末尾剥掉 followup 样本后，
    /// 再调 `clear_pending_followups` 清空。
    pub fn pending_followups_count(&self) -> usize {
        self.pending_followups.lock().len()
    }

    /// 清空假想分支记录（Python 侧对齐后调用）。
    pub fn clear_pending_followups(&self) {
        self.pending_followups.lock().clear();
    }

    /// 是否有假想分支（Pass 3 用：有则 ReplayEngine 需剥离 followup 样本）。
    pub fn has_pending_followups(&self) -> bool {
        !self.pending_followups.lock().is_empty()
    }
}

impl BatchAgent for MortalBatchAgent {
    #[inline]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[inline]
    fn oracle_obs_version(&self) -> Option<u32> {
        self.is_oracle.then_some(self.version)
    }

    fn set_scene(
        &mut self,
        index: usize,
        _: &[EventExt],
        state: &PlayerState,
        invisible_state: Option<InvisibleState>,
    ) -> Result<()> {
        self.evaluated = false;
        let cans = state.last_cans();

        if self.enable_quick_eval
            && cans.can_discard
            && !cans.can_riichi
            && !cans.can_tsumo_agari
            && !cans.can_ankan
            && !cans.can_kakan
            && !cans.can_ryukyoku
        {
            let candidates = state.discard_candidates_aka();
            let mut only_candidate = None;
            for (tile, &flag) in candidates.iter().enumerate() {
                if !flag {
                    continue;
                }
                match only_candidate.take() {
                    None => only_candidate = Some(tile),
                    Some(_) => break,
                }
            }

            if let Some(tile_id) = only_candidate {
                let actor = self.player_ids[index];
                let pai = must_tile!(tile_id);
                let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
                let ev = Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                };
                self.quick_eval_reactions[index] = Some(ev);
                return Ok(());
            }
        }

        let need_kan_select = if !cans.can_ankan && !cans.can_kakan {
            false
        } else if !self.enable_quick_eval {
            true
        } else {
            state.ankan_candidates().len() + state.kakan_candidates().len() > 1
        };

        // ── skip_encode 模式（Pass3 重放）：不编码 obs，构造等长占位批
        // （dummy obs）写入 states，让后续 evaluate→react_batch 照常
        // 消费 ReplayEngine 缓存。占位批长度 = 本决策点的样本数。
        // get_reaction 零改动（通过 pending_followups 生成 meta.followup）。
        if self.skip_encode {
            // 占位 obs（全零，ReplayEngine 不读内容）
            let shape = obs_shape(self.version);
            let dummy_obs = ndarray::Array2::<f32>::zeros(shape);
            let dummy_mask = ndarray::Array1::<bool>::from_elem(shape.1, false);
            // 与编码路径同源：真实收集假想分支（判定可执行性），
            // 样本数 / label / 顺序必然与编码路径完全一致。
            let followups = collect_followups(state);
            let mut sf = self.sync_fields.lock();
            // 写入等量占位样本（顺序: kan?→主, 其余算 followup）
            let base = sf.states.len();
            if need_kan_select {
                sf.kan_action_idxs[index] = Some(base);
                sf.states.push(dummy_obs.clone());
                sf.masks.push(dummy_mask.clone());
            }
            sf.action_idxs[index] = sf.states.len();
            sf.states.push(dummy_obs.clone());
            sf.masks.push(dummy_mask.clone());
            // followup 占位：数量对齐，pending_followups 记 (label, idx)
            // 供 get_reaction 按首推动作 label 生成 meta.followup。
            if !followups.is_empty() {
                let mut fups = self.pending_followups.lock();
                fups.clear();
                for (label, _) in followups {
                    fups.push((label, sf.states.len()));
                    sf.states.push(dummy_obs.clone());
                    sf.masks.push(dummy_mask.clone());
                }
            }
            return Ok(());
        }

        let version = self.version;
        let state = state.clone();

        // record_mode：分配本条 record 的槽位（决策点顺序），rayon 任务按
        // record_id 写入槽位，flush_encode 按序读取，避免并发 push 乱序。
        let record_slot = if self.record_mode {
            let rid = self.next_record;
            self.next_record += 1;
            let slots = self.record_slots.as_ref().unwrap();
            let mut slots = slots.lock();
            // 预分配占位（None=未完成），任务完成后置 Some
            slots.push(None);
            Some((rid, Arc::clone(&self.record_slots.as_ref().unwrap())))
        } else {
            None
        };

        let sync_fields = Arc::clone(&self.sync_fields);
        let wg = self.wg.clone();
        let pending_followups = Arc::clone(&self.pending_followups);
        rayon::spawn(move || {
            let _wg = wg;

            // Encode features in parallel within the game batch to utilize
            // multiple cores, as this can be very CPU-intensive, especially for
            // the sp feature (since v4).
            let kan = need_kan_select.then(|| state.encode_obs(version, true));
            let (feature, mask) = state.encode_obs(version, false);

            // 假想分支（review 用）：reach / chi(low/mid/high) / pon 各编一条
            // "假设执行该动作后的切牌局面" obs。只编码、不推理选择，
            // get_reaction 按首推动作取用对应分支，其余丢弃。
            let followups = collect_followups(&state);
            let followup_encoded: Vec<(usize, (Array2<f32>, Array1<bool>))> = followups
                .into_iter()
                .map(|(label, s)| (label, s.encode_obs(version, false)))
                .collect();

            // record_mode：按槽位写回（决策点顺序保证），不走共享 states。
            if let Some((rid, slots)) = record_slot {
                let mut samples: Vec<(Array2<f32>, Array1<bool>)> = Vec::new();
                if let Some((kan_feature, kan_mask)) = kan {
                    samples.push((kan_feature, kan_mask));
                }
                samples.push((feature, mask));
                for (_, (f, m)) in followup_encoded {
                    samples.push((f, m));
                }
                slots.lock()[rid] = Some(samples);
                return;
            }

            let SyncFields {
                states,
                invisible_states,
                masks,
                action_idxs,
                kan_action_idxs,
            } = &mut *sync_fields.lock();
            if let Some((kan_feature, kan_mask)) = kan {
                kan_action_idxs[index] = Some(states.len());
                states.push(kan_feature);
                masks.push(kan_mask);
                if let Some(invisible_state) = invisible_state.clone() {
                    invisible_states.push(invisible_state);
                }
            }

            action_idxs[index] = states.len();
            states.push(feature);
            masks.push(mask);
            if let Some(invisible_state) = invisible_state {
                invisible_states.push(invisible_state);
            }

            // 假想分支排在主决策样本之后，get_reaction 取用后由
            // consume_pending_followups 在 Python 侧对齐跳过。
            if !followup_encoded.is_empty() {
                let mut fups = pending_followups.lock();
                fups.clear();
                for (label, (f, m)) in followup_encoded {
                    fups.push((label, states.len()));
                    states.push(f);
                    masks.push(m);
                    // 假想分支不编 invisible（review 场景 is_oracle=false），
                    // 若将来在 oracle 模式下使用需在此处补 invisible_state。
                }
            }
        });

        Ok(())
    }

    fn get_reaction(
        &mut self,
        index: usize,
        _: &[EventExt],
        state: &PlayerState,
        _: Option<InvisibleState>,
    ) -> Result<EventExt> {
        // record_mode（Pass 1 录制）：不 evaluate、不等编码，返回占位动作。
        // 编码任务已在 set_scene 提交到 rayon 池并行；Pass 1 结束后由
        // flush_encode() 一次性等完。reaction 本身被 server 丢弃，占位即可。
        if self.record_mode {
            return Ok(EventExt::no_meta(placeholder_event(state)?));
        }

        if self.enable_quick_eval
            && let Some(ev) = self.quick_eval_reactions[index].take()
        {
            return Ok(EventExt::no_meta(ev));
        }

        if !self.evaluated {
            self.evaluate()?;
            self.evaluated = true;
        }
        let start = Instant::now();

        let mut sync_fields = self.sync_fields.lock();
        let action_idx = sync_fields.action_idxs[index];
        let kan_select_idx = sync_fields.kan_action_idxs[index].take();

        let actor = self.player_ids[index];
        let akas_in_hand = state.akas_in_hand();
        let cans = state.last_cans();

        let orig_action = self.actions[action_idx];
        let action =
            if self.enable_rule_based_agari_guard && orig_action == 43 && !state.rule_based_agari()
            {
                // The engine wants agari, but the rule-based engine is against
                // it. In rule-based agari guard mode, it will force to execute
                // the best alternative option other than agari.
                let mut q_values = self.q_values[action_idx];
                q_values[43] = f32::MIN;
                q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, l), (_, r)| l.total_cmp(r))
                    .unwrap()
                    .0
            } else {
                orig_action
            };

        let event = match action {
            0..=36 => {
                ensure!(
                    cans.can_discard,
                    "failed discard check: {}",
                    state.brief_info()
                );

                let pai = must_tile!(action);
                let tsumogiri = state.last_self_tsumo().is_some_and(|t| t == pai);
                Event::Dahai {
                    actor,
                    pai,
                    tsumogiri,
                }
            }

            37 => {
                ensure!(
                    cans.can_riichi,
                    "failed riichi check: {}",
                    state.brief_info()
                );

                Event::Reach { actor }
            }

            38 => {
                ensure!(
                    cans.can_chi_low,
                    "failed chi low check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;
                let first = pai.next();

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(3m) | tu8!(4m) => akas_in_hand[0],
                    tu8!(3p) | tu8!(4p) => akas_in_hand[1],
                    tu8!(3s) | tu8!(4s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [first.akaize(), first.next().akaize()]
                } else {
                    [first, first.next()]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }
            39 => {
                ensure!(
                    cans.can_chi_mid,
                    "failed chi mid check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(4m) | tu8!(6m) => akas_in_hand[0],
                    tu8!(4p) | tu8!(6p) => akas_in_hand[1],
                    tu8!(4s) | tu8!(6s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [pai.prev().akaize(), pai.next().akaize()]
                } else {
                    [pai.prev(), pai.next()]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }
            40 => {
                ensure!(
                    cans.can_chi_high,
                    "failed chi high check: {}",
                    state.brief_info()
                );

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;
                let last = pai.prev();

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(6m) | tu8!(7m) => akas_in_hand[0],
                    tu8!(6p) | tu8!(7p) => akas_in_hand[1],
                    tu8!(6s) | tu8!(7s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [last.prev().akaize(), last.akaize()]
                } else {
                    [last.prev(), last]
                };
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }

            41 => {
                ensure!(cans.can_pon, "failed pon check: {}", state.brief_info());

                let pai = state
                    .last_kawa_tile()
                    .context("invalid state: no last kawa tile")?;

                let can_akaize_consumed = match pai.as_u8() {
                    tu8!(5m) => akas_in_hand[0],
                    tu8!(5p) => akas_in_hand[1],
                    tu8!(5s) => akas_in_hand[2],
                    _ => false,
                };
                let consumed = if can_akaize_consumed {
                    [pai.akaize(), pai.deaka()]
                } else {
                    [pai.deaka(); 2]
                };
                Event::Pon {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed,
                }
            }

            42 => {
                ensure!(
                    cans.can_daiminkan || cans.can_ankan || cans.can_kakan,
                    "failed kan check: {}",
                    state.brief_info()
                );

                let ankan_candidates = state.ankan_candidates();
                let kakan_candidates = state.kakan_candidates();

                let tile = if let Some(kan_idx) = kan_select_idx {
                    let tile = must_tile!(self.actions[kan_idx]);
                    ensure!(
                        ankan_candidates.contains(&tile) || kakan_candidates.contains(&tile),
                        "kan choice not in kan candidates: {}",
                        state.brief_info()
                    );
                    tile
                } else if cans.can_daiminkan {
                    state
                        .last_kawa_tile()
                        .context("invalid state: no last kawa tile")?
                } else if cans.can_ankan {
                    ankan_candidates[0]
                } else {
                    kakan_candidates[0]
                };

                if cans.can_daiminkan {
                    let consumed = if tile.is_aka() {
                        [tile.deaka(); 3]
                    } else {
                        [tile.akaize(), tile, tile]
                    };
                    Event::Daiminkan {
                        actor,
                        target: cans.target_actor,
                        pai: tile,
                        consumed,
                    }
                } else if cans.can_ankan && ankan_candidates.contains(&tile.deaka()) {
                    Event::Ankan {
                        actor,
                        consumed: [tile.akaize(), tile, tile, tile],
                    }
                } else {
                    let can_akaize_target = match tile.as_u8() {
                        tu8!(5m) => akas_in_hand[0],
                        tu8!(5p) => akas_in_hand[1],
                        tu8!(5s) => akas_in_hand[2],
                        _ => false,
                    };
                    let (pai, consumed) = if can_akaize_target {
                        (tile.akaize(), [tile.deaka(); 3])
                    } else {
                        (tile.deaka(), [tile.akaize(), tile.deaka(), tile.deaka()])
                    };
                    Event::Kakan {
                        actor,
                        pai,
                        consumed,
                    }
                }
            }

            43 => {
                ensure!(
                    cans.can_agari(),
                    "failed hora check: {}",
                    state.brief_info(),
                );

                Event::Hora {
                    actor,
                    target: cans.target_actor,
                    deltas: None,
                    ura_markers: None,
                }
            }

            44 => {
                ensure!(
                    cans.can_ryukyoku,
                    "failed ryukyoku check: {}",
                    state.brief_info()
                );

                Event::Ryukyoku { deltas: None }
            }

            // 45
            _ => Event::None,
        };

        let mut meta = self.gen_meta(state, action_idx);

        // 假想分支（review 用）：首推动作（guard 修正后）为 reach/chi/pon 时，
        // 从对应假想样本的 q_values 取 mask 内 argmax 作为"执行后切牌第一选"。
        {
            let followups = self.pending_followups.lock();
            if !followups.is_empty() {
                // 首推动作对应的假想分支 label；37=reach, 38~40=chi, 41=pon
                let wanted = match action {
                    37..=41 => Some(action),
                    _ => None,
                };
                if let Some(label) = wanted {
                    if let Some(&(_, fup_idx)) = followups.iter().find(|(l, _)| *l == label) {
                        let fup_q = self.q_values[fup_idx];
                        let fup_mask = self.masks_recv[fup_idx];
                        let best = fup_q
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| fup_mask[*i])
                            .max_by(|(_, l), (_, r)| l.total_cmp(r))
                            .map(|(i, _)| i);
                        // 假想局面一定是切牌决策（mask 仅 0..=36），
                        // 多个可切牌时才返回（单切牌局面玩家无选择，无需展示）。
                        let n_cands = fup_mask[..37].iter().filter(|&&m| m).count();
                        if let Some(best_idx) = best.filter(|_| n_cands > 1) {
                            let pai = must_tile!(best_idx);
                            let tsumogiri = followup_tsumogiri(label, state, pai);
                            meta.followup = Some(vec![Event::Dahai {
                                actor,
                                pai,
                                tsumogiri,
                            }]);
                        }
                    }
                }
            }
        }

        let eval_time_ns = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO)
            .saturating_add(self.last_eval_elapsed)
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);

        meta.eval_time_ns = Some(eval_time_ns);
        meta.batch_size = Some(self.last_batch_size);
        meta.kan_select = kan_select_idx.map(|kan_idx| Box::new(self.gen_meta(state, kan_idx)));

        Ok(EventExt {
            event,
            meta: Some(meta),
        })
    }
}
