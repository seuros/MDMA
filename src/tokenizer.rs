//! GGUF tokenization: SentencePiece, Gemma 4 BPE, and byte-level BPE.
use crate::gguf::GgufModel;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

const SPM_SPACE: char = '\u{2581}';

mod byte_bpe;
mod chat;
pub use chat::{ChatFormat, Markers};

pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    piece_to_id: HashMap<String, u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    add_bos: bool,
    add_space_prefix: bool,
    merge_ranks: Option<HashMap<String, HashMap<String, usize>>>,
    specials: Vec<(String, u32)>,
    stop_ids: Vec<u32>,
    byte_bpe: Option<tokenizers::pipeline::PipelineTokenizer>,
    chat: Option<ChatFormat>,
}

struct Sym {
    text: String,
    prev: Option<usize>,
    next: Option<usize>,
}

struct Bigram {
    left: usize,
    right: usize,
    score: f32,
    bytes: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: highest score first; on a tie, smaller `left` first.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl Tokenizer {
    pub fn from_gguf(model: &GgufModel) -> Option<Self> {
        let kind = model.metadata_str("tokenizer.ggml.model")?;
        if !matches!(kind, "llama" | "gemma4" | "gpt2") {
            return None;
        }
        let pieces = model.metadata_string_array("tokenizer.ggml.tokens")?;
        let scores = if kind == "gpt2" && !model.has_metadata("tokenizer.ggml.scores") {
            vec![0.0; pieces.len()]
        } else {
            model.metadata_f32_array("tokenizer.ggml.scores")?
        };
        if pieces.is_empty()
            || pieces.len() != scores.len()
            || scores.iter().any(|s| !s.is_finite())
        {
            return None;
        }
        let merge_ranks = if kind == "gemma4" {
            let mut ranks = HashMap::<String, HashMap<String, usize>>::new();
            for (rank, pair) in model
                .metadata_string_array("tokenizer.ggml.merges")?
                .into_iter()
                .enumerate()
            {
                let (a, b) = pair.split_once(' ')?;
                ranks
                    .entry(a.to_owned())
                    .or_default()
                    .insert(b.to_owned(), rank);
            }
            Some(ranks)
        } else {
            None
        };
        let types = model.metadata_i32_array("tokenizer.ggml.token_type");
        if model.has_metadata("tokenizer.ggml.token_type") && types.as_ref()?.len() != pieces.len()
        {
            return None;
        }
        if kind == "gpt2" {
            types.as_ref()?;
            for suffix in [
                "bos_token_id",
                "eos_token_id",
                "eot_token_id",
                "eom_token_id",
                "unknown_token_id",
            ] {
                let key = format!("tokenizer.ggml.{suffix}");
                if model.has_metadata(&key) && model.metadata_u32(&key)? as usize >= pieces.len() {
                    return None;
                }
            }
        }
        let mut specials: Vec<_> = pieces
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                types
                    .as_ref()
                    .and_then(|t| t.get(*i))
                    .is_some_and(|t| matches!(t, 3 | 4))
                    || (kind == "gemma4"
                        && matches!(p.as_str(), "<eos>" | "<turn|>" | "<|tool_response>"))
            })
            .map(|(i, p)| (p.clone(), i as u32))
            .filter(|(p, _)| !p.is_empty())
            .collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let mut stop_ids: Vec<_> = model
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .into_iter()
            .collect();
        if kind == "gemma4" {
            for name in ["<eos>", "<turn|>", "<|tool_response>"] {
                if let Some(id) = pieces.iter().position(|p| p == name) {
                    stop_ids.push(id as u32);
                }
            }
        }
        if kind == "gpt2" {
            for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
                stop_ids.extend(model.metadata_u32(key));
            }
            if let Some(id) = pieces.iter().position(|p| p == "<|endoftext|>") {
                stop_ids.push(id as u32);
            }
        }
        let byte_bpe = if kind == "gpt2" {
            Some(byte_bpe::from_gguf(model, &pieces, &specials)?)
        } else {
            None
        };
        let piece_to_id: HashMap<String, u32> = pieces
            .iter()
            .enumerate()
            .map(|(id, p)| (p.clone(), id as u32))
            .collect();
        let chat = ChatFormat::detect(model.metadata_str("tokenizer.chat_template"), |marker| {
            piece_to_id.contains_key(marker)
        });

        Some(Self {
            pieces,
            scores,
            piece_to_id,
            bos_id: model.metadata_u32("tokenizer.ggml.bos_token_id"),
            eos_id: model.metadata_u32("tokenizer.ggml.eos_token_id"),
            unk_id: model.metadata_u32("tokenizer.ggml.unknown_token_id"),
            add_bos: model
                .metadata_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(kind != "gpt2"),
            add_space_prefix: model
                .metadata_bool("tokenizer.ggml.add_space_prefix")
                .unwrap_or(true),
            merge_ranks,
            specials,
            stop_ids,
            byte_bpe,
            chat,
        })
    }

    /// The model's chat format, if recognised.
    pub const fn chat_format(&self) -> Option<ChatFormat> {
        self.chat
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// The ids of every vocabulary piece `matches` accepts, in id order.
    pub fn ids_where(&self, matches: impl Fn(&str) -> bool) -> Vec<u32> {
        (0..)
            .zip(&self.pieces)
            .filter(|(_, piece)| matches(piece))
            .map(|(id, _)| id)
            .collect()
    }

    pub fn is_eos(&self, id: u32) -> bool {
        Some(id) == self.eos_id || self.stop_ids.contains(&id)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.try_encode(text)
            .expect("tokenizer encoding failed; use try_encode for fallible input")
    }

    /// Fallible encoding for request handlers and untrusted input.
    pub fn try_encode(&self, text: &str) -> crate::backend::Result<Vec<u32>> {
        let mut out = Vec::new();
        if self.add_bos {
            out.extend(self.bos_id);
        }
        if let Some(bpe) = &self.byte_bpe {
            let mut encoded = Vec::new();
            bpe.encode_into(
                text,
                &tokenizers::pipeline::EncodeOptions::no_specials(),
                &mut encoded,
            )
            .map_err(|e| crate::backend::Error::Backend(format!("tokenizer encoding: {e}")))?;
            out.extend(encoded.into_iter().map(u32::from));
            return Ok(out);
        }
        let mut remaining = text;
        let mut prefix = self.add_space_prefix;
        while !remaining.is_empty() {
            let special = self
                .specials
                .iter()
                .filter_map(|(piece, id)| remaining.find(piece).map(|at| (at, piece, *id)))
                .min_by_key(|(at, _, _)| *at);
            if let Some((at, piece, id)) = special {
                if at > 0 {
                    out.extend(self.encode_plain(&remaining[..at], prefix));
                }
                out.push(id);
                remaining = &remaining[at + piece.len()..];
                // SentencePiece restarts its space prefix after a control
                // token; Gemma's BPE does not insert that artificial space.
                prefix = self.add_space_prefix && self.merge_ranks.is_none();
            } else {
                out.extend(self.encode_plain(remaining, prefix));
                break;
            }
        }
        Ok(out)
    }

    /// Byte BPE may defer text until a split UTF-8 sequence is complete.
    pub fn stream_decoder(&self) -> impl FnMut(u32) -> crate::backend::Result<String> + '_ {
        let mut stream = self.byte_bpe.as_ref().map(|bpe| bpe.decode_stream(false));
        move |id| match &mut stream {
            Some(stream) => stream
                .step(id)
                .map(std::option::Option::unwrap_or_default)
                .map_err(|e| crate::backend::Error::Backend(format!("tokenizer decoding: {e}"))),
            None => Ok(self.decode(&[id])),
        }
    }

    fn encode_plain(&self, text: &str, prefix: bool) -> Vec<u32> {
        let mut normalized = String::new();
        if prefix {
            normalized.push(SPM_SPACE);
        }
        for c in text.chars() {
            normalized.push(if c == ' ' { SPM_SPACE } else { c });
        }

        if self.merge_ranks.is_some() {
            let mut ids = Vec::new();
            let mut start = 0;
            while start < normalized.len() {
                let newline = normalized.as_bytes()[start] == b'\n';
                let end = normalized[start..]
                    .char_indices()
                    .find(|(_, c)| (*c == '\n') != newline)
                    .map_or(normalized.len(), |(i, _)| start + i);
                let part = &normalized[start..end];
                if newline && let Some(&id) = self.piece_to_id.get(part) {
                    ids.push(id);
                } else {
                    ids.extend(self.spm_merge(part));
                }
                start = end;
            }
            ids
        } else {
            self.spm_merge(&normalized)
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(bpe) = &self.byte_bpe {
            let ids: Vec<_> = ids
                .iter()
                .copied()
                .filter(|id| (*id as usize) < self.pieces.len())
                .collect();
            return bpe
                .decode(&ids, false)
                .expect("validated byte BPE vocabulary");
        }
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(piece) = self.pieces.get(id as usize) else {
                continue;
            };
            match byte_token_value(piece) {
                Some(b) => bytes.push(b),
                None => bytes.extend(piece.replace(SPM_SPACE, " ").into_bytes()),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn spm_merge(&self, normalized: &str) -> Vec<u32> {
        let mut syms: Vec<Sym> = normalized
            .chars()
            .map(|c| Sym {
                text: c.to_string(),
                prev: None,
                next: None,
            })
            .collect();
        let n = syms.len();
        for (i, sym) in syms.iter_mut().enumerate() {
            sym.prev = i.checked_sub(1);
            sym.next = (i + 1 < n).then_some(i + 1);
        }
        if n == 0 {
            return Vec::new();
        }

        let mut heap = BinaryHeap::new();
        for i in 0..n - 1 {
            self.try_push_bigram(&mut heap, &syms, i, i + 1);
        }

        while let Some(bigram) = heap.pop() {
            if syms[bigram.left].next != Some(bigram.right) {
                continue;
            }
            if syms[bigram.left].text.is_empty() || syms[bigram.right].text.is_empty() {
                continue;
            }
            if syms[bigram.left].text.len() + syms[bigram.right].text.len() != bigram.bytes {
                continue;
            }

            let right_text = std::mem::take(&mut syms[bigram.right].text);
            syms[bigram.left].text.push_str(&right_text);
            let right_next = syms[bigram.right].next;
            syms[bigram.left].next = right_next;
            if let Some(rn) = right_next {
                syms[rn].prev = Some(bigram.left);
            }

            if let Some(p) = syms[bigram.left].prev {
                self.try_push_bigram(&mut heap, &syms, p, bigram.left);
            }
            if let Some(next) = syms[bigram.left].next {
                self.try_push_bigram(&mut heap, &syms, bigram.left, next);
            }
        }

        let mut ids = Vec::new();
        let mut cur = Some(0usize);
        while let Some(i) = cur {
            self.piece_to_ids(&syms[i].text, &mut ids);
            cur = syms[i].next;
        }
        ids
    }

    fn try_push_bigram(
        &self,
        heap: &mut BinaryHeap<Bigram>,
        syms: &[Sym],
        left: usize,
        right: usize,
    ) {
        let merged = format!("{}{}", syms[left].text, syms[right].text);
        if let Some(&id) = self.piece_to_id.get(&merged) {
            let score = if let Some(ranks) = &self.merge_ranks {
                let Some(rank) = ranks
                    .get(syms[left].text.as_str())
                    .and_then(|right_ranks| right_ranks.get(syms[right].text.as_str()))
                else {
                    return;
                };
                -(*rank as f32)
            } else {
                self.scores[id as usize]
            };
            heap.push(Bigram {
                left,
                right,
                score,
                bytes: merged.len(),
            });
        }
    }

    fn piece_to_ids(&self, piece: &str, out: &mut Vec<u32>) {
        if let Some(&id) = self.piece_to_id.get(piece) {
            out.push(id);
            return;
        }
        for b in piece.bytes() {
            let byte_tok = format!("<0x{b:02X}>");
            match self.piece_to_id.get(&byte_tok) {
                Some(&id) => out.push(id),
                None => out.extend(self.unk_id),
            }
        }
    }
}

fn byte_token_value(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
pub(crate) mod tests;
