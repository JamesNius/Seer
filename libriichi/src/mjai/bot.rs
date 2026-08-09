use super::EventWithCanAct;
use super::{Event, EventExt};
use crate::agent::{BatchAgent, MortalBatchAgent};
use crate::state::PlayerState;

use anyhow::{Context, Result};
use pyo3::prelude::*;
use serde_json as json;

#[pyclass]
pub struct Bot {
    agent: MortalBatchAgent,
    state: PlayerState,
    log: Vec<EventExt>,
}

#[pymethods]
impl Bot {
    #[new]
    fn new(engine: PyObject, player_id: u8) -> Result<Self> {
        let agent = MortalBatchAgent::new(engine, &[player_id])?;
        let state = PlayerState::new(player_id);
        Ok(Self {
            agent,
            state,
            log: vec![],
        })
    }

    /// Returns the reaction to `line`, if it can react, `None` otherwise.
    ///
    /// Set `can_act` or `line_json['can_act']` to `False` to force the bot to
    /// only update its state without making any reaction.
    ///
    /// Both `line` and the return value are JSON strings representing one
    /// single mjai event.
    #[pyo3(name = "react")]
    #[pyo3(signature = (line, /, *, can_act=true))]
    fn react_py(&mut self, line: &str, can_act: bool, py: Python<'_>) -> Result<Option<String>> {
        py.allow_threads(move || self.react(line, can_act))
    }

    /// 返回 agent 当前待取用的假想分支样本数（不清空）。
    /// review server Pass 1 按此对齐 record 的 followup 样本数。
    #[pyo3(name = "pending_followups_count")]
    fn pending_followups_count_py(&self) -> usize {
        self.agent.pending_followups_count()
    }

    /// 清空假想分支记录（review server Pass 3 对齐后调用）。
    #[pyo3(name = "clear_pending_followups")]
    fn clear_pending_followups_py(&self) {
        self.agent.clear_pending_followups();
    }

    /// 是否有假想分支（Pass 3 用：有则 ReplayEngine 需剥离 followup 样本）。
    #[pyo3(name = "has_pending_followups")]
    fn has_pending_followups_py(&self) -> bool {
        self.agent.has_pending_followups()
    }

    /// record_mode 结束（Pass 1 录完）后调用：等所有编码任务完成，
    /// 返回按 record 分组的 (obs, masks, invisible) 列表供 Pass 2 批量推理。
    #[pyo3(name = "flush_encode")]
    fn flush_encode_py(&mut self) -> Result<Py<PyAny>> {
        self.agent.flush_encode()
    }
}

impl Bot {
    fn react(&mut self, line: &str, can_act: bool) -> Result<Option<String>> {
        let data: EventWithCanAct =
            json::from_str(line).with_context(|| format!("failed to parse event {line}"))?;

        match data.event {
            Event::StartGame { .. } => {
                self.agent.start_game(0)?;
            }
            Event::EndKyoku => {
                self.log.clear();
                self.agent.end_kyoku(0)?;
            }
            Event::EndGame => {
                self.agent.end_game(0, &Default::default())?;
            }
            _ => {
                self.log.push(EventExt::no_meta(data.event.clone()));
            }
        };

        let cans = self.state.update(&data.event)?;
        if !can_act || matches!(data.can_act, Some(false)) || !cans.can_act() {
            return Ok(None);
        }

        self.agent
            .set_scene(0, &self.log, &self.state, None)
            .context("failed to add state")?;
        let reaction = self
            .agent
            .get_reaction(0, &self.log, &self.state, None)
            .context("failed to get reaction")?;

        let ret = json::to_string(&reaction)?;
        Ok(Some(ret))
    }
}
