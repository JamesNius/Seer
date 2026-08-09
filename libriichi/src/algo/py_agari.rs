//! Read-only Python bindings for agari (和了点数) calculation.
//!
//! This module exposes a pure-function entry point to compute the nominal ron
//! value of a winning hand, mirroring `state::agent_helper::agari_points` but
//! taking every component explicitly so it can be driven from a replayed log
//! (used by the offline value-distribution extractor). It is purely additive:
//! no existing logic is touched.

use crate::algo::agari::AgariCalculator;
use crate::py_helper::add_submodule;
use crate::tile::Tile;

use std::str::FromStr;

use anyhow::{Context, Result};
use pyo3::prelude::*;
use tinyvec::ArrayVec;

fn parse_tiles(list: &[String], what: &str) -> Result<Vec<Tile>> {
    list.iter()
        .map(|s| Tile::from_str(s).with_context(|| format!("invalid {what} tile: {s}")))
        .collect()
}

/// Compute the nominal ron value (and tsumo-equivalent) of a winning hand.
///
/// All tile arguments are mjai pai strings (e.g. "5m", "5mr", "E"). Returns a
/// tuple `(ron, tsumo_total, han, fu, doras)`:
///
/// * `ron` — the nominal ron score of the hand (the value-distribution target).
/// * `tsumo_total` — the total points collected if this were a tsumo win, used
///   to convert tsumo wins into their ron-equivalent value.
/// * `han`, `fu` — for sanity checking / debugging.
/// * `doras` — total dora count (手牌+副露+杠+赤+里宝 when riichi)。
///
/// * `tehai` — the closed hand **including** the winning tile (3n+2 tiles).
/// * `winning_tile` — the tile that completed the hand (may be aka).
/// * `is_ron` — ron vs tsumo.
/// * `dora_indicators` — revealed dora indicators (表ドラ).
/// * `ura_indicators` — revealed ura indicators (里ドラ); only counted when
///   `riichi` holds.
/// * `chis`, `pons`, `minkans`, `ankans` — the winner's open melds. `chis` /
///   `pons` are given as the full tile list of each meld (used for dora
///   counting); `minkans` / `ankans` as the single (deaka'd) tile of each kan.
/// * `additional_hans` — circumstantial yaku han (riichi, ippatsu, menzen
///   tsumo, haitei, houtei, rinshan, chankan, ...).
#[pyfunction]
#[pyo3(signature = (
    tehai,
    winning_tile,
    is_ron,
    bakaze,
    jikaze,
    is_oya,
    additional_hans,
    dora_indicators,
    ura_indicators,
    chis,
    pons,
    minkans,
    ankans,
    riichi,
))]
#[allow(clippy::too_many_arguments)]
pub fn calc_agari_point(
    tehai: Vec<String>,
    winning_tile: &str,
    is_ron: bool,
    bakaze: u8,
    jikaze: u8,
    is_oya: bool,
    additional_hans: u8,
    dora_indicators: Vec<String>,
    ura_indicators: Vec<String>,
    chis: Vec<Vec<String>>,
    pons: Vec<Vec<String>>,
    minkans: Vec<String>,
    ankans: Vec<String>,
    riichi: bool,
) -> Result<(i32, i32, u8, u8, u8)> {
    let win_tile = Tile::from_str(winning_tile).context("invalid winning tile")?;

    // Build the tehai count array (includes the winning tile already).
    let tehai_tiles = parse_tiles(&tehai, "tehai")?;
    let mut tehai_cnt = [0u8; 34];
    for t in &tehai_tiles {
        tehai_cnt[t.deaka().as_usize()] += 1;
    }
    anyhow::ensure!(
        tehai_cnt.iter().map(|&c| c as usize).sum::<usize>() % 3 == 2,
        "tehai must have 3n+2 tiles (winning tile included)"
    );

    // Dora factor per tile id, accumulated across every indicator.
    let mut dora_factor = [0u8; 34];
    let mut doras = 0u8;
    let dora_inds = parse_tiles(&dora_indicators, "dora indicator")?;
    for ind in &dora_inds {
        dora_factor[ind.next().as_usize()] += 1;
    }

    // Count dora in the closed hand.
    for (id, &cnt) in tehai_cnt.iter().enumerate() {
        doras += cnt * dora_factor[id];
    }
    // Aka dora in the closed hand.
    doras += tehai_tiles.iter().filter(|t| t.is_aka()).count() as u8;

    // Count dora in open melds. chi / pon are given as full tile lists.
    let mut meld_aka = 0u8;
    let mut add_meld = |tiles: &[String]| -> Result<()> {
        let ts = parse_tiles(tiles, "meld")?;
        for t in &ts {
            let id = t.deaka().as_usize();
            doras += dora_factor[id];
            if t.is_aka() {
                meld_aka += 1;
            }
        }
        Ok(())
    };
    let mut chis_id: ArrayVec<[u8; 4]> = ArrayVec::new();
    let mut pons_id: ArrayVec<[u8; 4]> = ArrayVec::new();
    for meld in &chis {
        add_meld(meld)?;
        // representative (deaka'd) tile of the chi, for agari's meld list.
        let ts = parse_tiles(meld, "chi")?;
        if let Some(first) = ts.first() {
            chis_id.push(first.deaka().as_u8());
        }
    }
    for meld in &pons {
        add_meld(meld)?;
        let ts = parse_tiles(meld, "pon")?;
        if let Some(first) = ts.first() {
            pons_id.push(first.deaka().as_u8());
        }
    }
    doras += meld_aka;

    // Kans are given as their single (deaka'd) tile; each contributes 4 copies.
    let minkan_tiles = parse_tiles(&minkans, "minkan")?;
    let ankan_tiles = parse_tiles(&ankans, "ankan")?;
    let mut minkans_id: ArrayVec<[u8; 4]> = ArrayVec::new();
    let mut ankans_id: ArrayVec<[u8; 4]> = ArrayVec::new();
    for t in &minkan_tiles {
        let id = t.deaka();
        doras += 4 * dora_factor[id.as_usize()];
        if t.is_aka() {
            doras += 1;
        }
        minkans_id.push(id.as_u8());
    }
    for t in &ankan_tiles {
        let id = t.deaka();
        doras += 4 * dora_factor[id.as_usize()];
        if t.is_aka() {
            doras += 1;
        }
        ankans_id.push(id.as_u8());
    }

    // Ura dora, only when the winner has an accepted riichi.
    if riichi {
        let ura_inds = parse_tiles(&ura_indicators, "ura indicator")?;
        for ura in &ura_inds {
            let next = ura.next();
            let mut count = tehai_cnt[next.as_usize()];
            // A kan of the ura tile contributes all 4 copies.
            if ankans_id.contains(&next.as_u8()) || minkans_id.contains(&next.as_u8()) {
                count += 4;
            }
            doras += count;
        }
    }

    let is_menzen = chis.is_empty() && pons.is_empty() && minkans.is_empty();
    let agari_calc = AgariCalculator {
        tehai: &tehai_cnt,
        is_menzen,
        chis: &chis_id,
        pons: &pons_id,
        minkans: &minkans_id,
        ankans: &ankans_id,
        bakaze,
        jikaze,
        winning_tile: win_tile.deaka().as_u8(),
        is_ron,
    };
    let agari = agari_calc
        .agari(additional_hans, doras)
        .context("not a hora hand")?;
    let point = agari.point(is_oya);

    let (han, fu) = match agari {
        crate::algo::agari::Agari::Normal { fu, han } => (han, fu),
        crate::algo::agari::Agari::Yakuman(n) => (13 * n as u8, 0),
    };

    Ok((point.ron, point.tsumo_total(is_oya), han, fu, doras))
}

pub(crate) fn register_module(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    let m = PyModule::new(py, "agari")?;
    m.add_function(wrap_pyfunction!(calc_agari_point, &m)?)?;
    add_submodule(py, prefix, super_mod, &m)
}
