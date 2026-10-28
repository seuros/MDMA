//! Bounded GGUF header decoding.
use super::{Error, GgmlDType, Result, TensorDesc, tensor_n_bytes};
use candle_core::quantized::gguf_file::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

const MAX_ITEMS: u64 = 1 << 30;
const MAX_DEPTH: usize = 64;

pub(super) struct Header {
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorDesc>,
    pub data_offset: u64,
}

struct Reader<R> {
    input: R,
    position: u64,
    file_len: u64,
    version: u32,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Parse(message.into())
}

impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.input.read_exact(&mut bytes)?;
        self.position += N as u64;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn length(&mut self) -> Result<u64> {
        if self.version == 1 {
            self.u32().map(u64::from)
        } else {
            self.u64()
        }
    }

    fn prefix_bytes(&self) -> u64 {
        if self.version == 1 { 4 } else { 8 }
    }

    fn count(&self, count: u64, minimum: u64) -> Result<usize> {
        if count > MAX_ITEMS || count > self.file_len.saturating_sub(self.position) / minimum {
            return Err(invalid("GGUF count exceeds format limit or remaining file"));
        }
        usize::try_from(count).map_err(|_| invalid("GGUF count exceeds address space"))
    }

    fn string(&mut self) -> Result<String> {
        let length = self.length()?;
        let length = self.count(length, 1)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| invalid("GGUF string allocation failed"))?;
        bytes.resize(length, 0);
        self.input.read_exact(&mut bytes)?;
        self.position += length as u64;
        String::from_utf8(bytes).map_err(|_| invalid("GGUF string is not UTF-8"))
    }

    fn minimum(&self, kind: u32) -> Result<u64> {
        match kind {
            0 | 1 | 7 => Ok(1),
            2 | 3 => Ok(2),
            4..=6 => Ok(4),
            8 => Ok(self.prefix_bytes()),
            9 => Ok(4 + self.prefix_bytes()),
            10..=12 => Ok(8),
            _ => Err(invalid(format!("unknown GGUF metadata type {kind}"))),
        }
    }

    fn value(&mut self, kind: u32, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return Err(invalid("GGUF value nesting depth exceeds 64"));
        }
        Ok(match kind {
            0 => Value::U8(self.bytes::<1>()?[0]),
            1 => Value::I8(i8::from_le_bytes(self.bytes()?)),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => Value::Bool(match self.bytes::<1>()?[0] {
                0 => false,
                1 => true,
                _ => return Err(invalid("GGUF boolean must be 0 or 1")),
            }),
            8 => Value::String(self.string()?),
            9 => {
                let child = self.u32()?;
                let length = self.length()?;
                let count = self.count(length, self.minimum(child)?)?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(count)
                    .map_err(|_| invalid("GGUF array allocation failed"))?;
                for _ in 0..count {
                    values.push(self.value(child, depth + 1)?);
                }
                Value::Array(values)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            _ => return Err(invalid(format!("unknown GGUF metadata type {kind}"))),
        })
    }
}

fn alignment(metadata: &HashMap<String, Value>) -> Result<u64> {
    let value = match metadata.get("general.alignment") {
        None => return Ok(32),
        Some(Value::U8(n)) => u64::from(*n),
        Some(Value::U16(n)) => u64::from(*n),
        Some(Value::U32(n)) => u64::from(*n),
        Some(Value::U64(n)) => *n,
        Some(Value::I8(n)) if *n > 0 => u64::from(n.cast_unsigned()),
        Some(Value::I16(n)) if *n > 0 => u64::from(n.cast_unsigned()),
        Some(Value::I32(n)) if *n > 0 => u64::from(n.cast_unsigned()),
        Some(Value::I64(n)) if *n > 0 => n.cast_unsigned(),
        _ => return Err(invalid("GGUF alignment must be a positive integer")),
    };
    if !value.is_power_of_two() || value > MAX_ITEMS {
        return Err(invalid("GGUF alignment must be a power of two <= 2^30"));
    }
    Ok(value)
}

pub(super) fn read(file: &mut File) -> Result<Header> {
    let file_len = file.metadata()?.len();
    file.seek(SeekFrom::Start(0))?;
    let mut reader = Reader {
        input: BufReader::new(file),
        position: 0,
        file_len,
        version: 0,
    };
    if !matches!(reader.u32()?, 0x46554747 | 0x47475546) {
        return Err(invalid("invalid GGUF magic"));
    }
    reader.version = reader.u32()?;
    if !(1..=3).contains(&reader.version) {
        return Err(invalid("unsupported GGUF version"));
    }
    let tensor_count = reader.length()?;
    let metadata_count = reader.length()?;
    let tensor_count = reader.count(tensor_count, reader.prefix_bytes() + 16)?;
    let metadata_count = reader.count(metadata_count, reader.prefix_bytes() + 5)?;
    let mut metadata = HashMap::new();
    for _ in 0..metadata_count {
        let name = reader.string()?;
        let kind = reader.u32()?;
        let value = reader.value(kind, 0)?;
        if metadata.insert(name, value).is_some() {
            return Err(invalid("duplicate GGUF metadata key"));
        }
    }
    let alignment = alignment(&metadata)?;
    let mut tensors = HashMap::new();
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let dimensions = reader.u32()?;
        if dimensions > 4 {
            return Err(invalid("GGUF tensor has more than four dimensions"));
        }
        let mut shape = Vec::with_capacity(dimensions as usize);
        for _ in 0..dimensions {
            let dim = reader.length()?;
            shape
                .push(usize::try_from(dim).map_err(|_| invalid("tensor dimension exceeds usize"))?);
        }
        shape.reverse();
        let dtype = GgmlDType::from_code(reader.u32()?)?;
        let offset = reader.u64()?;
        if !offset.is_multiple_of(alignment) {
            return Err(invalid(format!("{name}: tensor offset is not aligned")));
        }
        let n_bytes = tensor_n_bytes(&name, dtype, &shape)?;
        if tensors
            .insert(
                name,
                TensorDesc {
                    dtype,
                    shape,
                    offset,
                    n_bytes,
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate GGUF tensor name"));
        }
    }
    let data_offset = reader
        .position
        .checked_add(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or_else(|| invalid("GGUF data offset overflows u64"))?;
    Ok(Header {
        metadata,
        tensors,
        data_offset,
    })
}

#[cfg(test)]
mod tests;
