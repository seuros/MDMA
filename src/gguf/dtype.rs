//! On-disk ggml types.
use candle_core::quantized::GgmlDType as Candle;

macro_rules! types {
    ($($name:ident = $code:literal),+ $(,)?) => {
        /// Weight storage format, with GGUF wire IDs as discriminants.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u32)]
        pub enum GgmlDType {
            $($name = $code,)+
            Q1_0 = 41,
        }

        impl GgmlDType {
            pub(super) fn from_code(code: u32) -> super::Result<Self> {
                match code {
                    $($code => Ok(Self::$name),)+
                    41 => Ok(Self::Q1_0),
                    _ => Err(super::Error::Parse(format!("unknown dtype for tensor {code}"))),
                }
            }

            /// Return the Candle equivalent, or `None` for Q1_0.
            pub const fn candle(self) -> Option<Candle> {
                match self {
                    $(Self::$name => Some(Candle::$name),)+
                    Self::Q1_0 => None,
                }
            }
        }

        impl From<Candle> for GgmlDType {
            fn from(dtype: Candle) -> Self {
                match dtype {
                    $(Candle::$name => Self::$name,)+
                }
            }
        }
    };
}

types! {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2K = 10, Q3K = 11, Q4K = 12, Q5K = 13,
    Q6K = 14, Q8K = 15, BF16 = 30,
}

impl GgmlDType {
    /// Bytes per block. Q1_0 stores an fp16 scale and 128 sign bits.
    pub fn type_size(self) -> usize {
        self.candle().map_or(18, |dtype| dtype.type_size())
    }

    pub fn block_size(self) -> usize {
        self.candle().map_or(128, |dtype| dtype.block_size())
    }
}

impl PartialEq<Candle> for GgmlDType {
    fn eq(&self, other: &Candle) -> bool {
        self.candle() == Some(*other)
    }
}

impl PartialEq<GgmlDType> for Candle {
    fn eq(&self, other: &GgmlDType) -> bool {
        other == self
    }
}
