use sha2::{Digest, Sha256};
use std::{
    fs::{File, Metadata},
    io::{self, Read, Seek},
    time::SystemTime,
};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Stamp {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    changed: (i64, i64),
}

impl Stamp {
    pub fn read(metadata: &Metadata) -> io::Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        if !metadata.is_file() {
            return Err(io::Error::other("model/cache must be a regular file"));
        }
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }

    pub const fn len(&self) -> u64 {
        self.len
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "callers pass freshly queried metadata"
    )]
    pub fn check(&self, metadata: Metadata) -> io::Result<()> {
        if *self != Self::read(&metadata)? {
            return Err(io::Error::other(
                "model/cache changed while reading; keep model files immutable",
            ));
        }
        Ok(())
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "callers pass freshly queried metadata"
    )]
    pub fn check_payload(&self, metadata: Metadata) -> io::Result<()> {
        let now = Self::read(&metadata)?;
        // An unlink changes the still-open inode's ctime, not its bytes: an
        // atomic replacement of a cache file, or the name a private copy was
        // created under. Per-tensor SHA also detects cache writes that
        // restore mtime.
        if self.len != now.len || self.modified != now.modified {
            return Err(io::Error::other("model/cache changed while reading"));
        }
        Ok(())
    }
}

pub(crate) fn digest(mut reader: impl Read) -> io::Result<String> {
    let mut hash = Sha256::new();
    let mut chunk = vec![0; 1024 * 1024];
    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hash.update(&chunk[..n]);
    }
    Ok(hex::encode(hash.finalize()))
}

pub(crate) fn fingerprint(file: &mut File) -> io::Result<(String, Stamp)> {
    let stamp = Stamp::read(&file.metadata()?)?;
    file.rewind()?;
    let sha = digest(&mut *file)?;
    stamp.check(file.metadata()?)?;
    Ok((sha, stamp))
}
