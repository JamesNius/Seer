use pyo3::prelude::*;

use std::ffi::CString;

pub(crate) fn add_submodule(
    py: Python<'_>,
    prefix: &str,
    super_mod: &Bound<'_, PyModule>,
    m: &Bound<'_, PyModule>,
) -> PyResult<()> {
    super_mod.add_submodule(m)?;

    let name = m.name()?;
    // prefix 可能是 "libriichi"（开发安装）或 "libriichi.libriichi"（wheel 安装）。
    // 两种都注册，保证 `import libriichi.consts` 在任何布局下可用。
    let top = prefix.split('.').next().unwrap_or(prefix);
    let script = CString::new(format!(
        "import sys; sys.modules['{prefix}.{name}'] = {name}; sys.modules.setdefault('{top}.{name}', {name}); del sys"
    ))?;
    py.run(&script, None, Some(&super_mod.dict()))
}
