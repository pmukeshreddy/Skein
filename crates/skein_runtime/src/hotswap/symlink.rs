//! Atomic symlink swap. Writes `LATEST.new -> new_target` then renames
//! it to `LATEST`. POSIX `rename(2)` is atomic on the same filesystem,
//! so concurrent readers see `LATEST` pointing to either the old or new
//! target — never broken.

use std::path::{Path, PathBuf};

use crate::error::RuntimeError;

pub fn atomic_symlink_swap(latest: &Path, new_target: &Path) -> Result<(), RuntimeError> {
    // Tempname includes the new target's basename to make `ls` interpretable
    // during a transient race.
    let parent = latest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let temp = parent.join(format!(
        "LATEST.new.{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    // Create the temporary symlink first.
    #[cfg(unix)]
    std::os::unix::fs::symlink(new_target, &temp).map_err(|source| RuntimeError::Io {
        path: temp.clone(),
        source,
    })?;
    #[cfg(not(unix))]
    return Err(RuntimeError::Io {
        path: temp.clone(),
        source: std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "atomic symlink swap is only implemented on Unix targets",
        ),
    });
    // POSIX `rename` is atomic when source and destination are on the
    // same filesystem (which they are, both being under `parent`).
    std::fs::rename(&temp, latest).map_err(|source| RuntimeError::Io {
        path: latest.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Read the current symlink target. Returns the path the symlink points to.
pub fn current_target(latest: &Path) -> Result<PathBuf, RuntimeError> {
    std::fs::read_link(latest).map_err(|source| RuntimeError::Io {
        path: latest.to_path_buf(),
        source,
    })
}
