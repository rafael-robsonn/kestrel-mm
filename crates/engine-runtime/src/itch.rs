//! Nasdaq TotalView-ITCH 5.0 order-level messages, big-endian, framed with the 2-byte
//! length prefix used by ITCH binary files and SoupBinTCP payloads.
//!
//! Prices on the wire are `u32` with 4 implied decimals and are rescaled to the internal
//! `10^-8` fixed point. Share counts are whole units.

use core_types::{EventBody, OrderId, Price, Qty, SCALE, Side, Timestamp};

pub const MSG_SYSTEM_EVENT: u8 = b'S';
pub const MSG_ADD: u8 = b'A';
pub const MSG_ADD_MPID: u8 = b'F';
pub const MSG_EXECUTED: u8 = b'E';
pub const MSG_EXECUTED_WITH_PRICE: u8 = b'C';
pub const MSG_CANCEL: u8 = b'X';
pub const MSG_DELETE: u8 = b'D';
pub const MSG_REPLACE: u8 = b'U';
pub const MSG_TRADE: u8 = b'P';

pub const LEN_ADD: usize = 36;
pub const LEN_ADD_MPID: usize = 40;
pub const LEN_EXECUTED: usize = 31;
pub const LEN_EXECUTED_WITH_PRICE: usize = 36;
pub const LEN_CANCEL: usize = 23;
pub const LEN_DELETE: usize = 19;
pub const LEN_REPLACE: usize = 35;
pub const LEN_TRADE: usize = 44;

const ITCH_PRICE_TO_RAW: i64 = SCALE / 10_000;
const MAX_TIMESTAMP: u64 = (1 << 48) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ItchError {
    #[error("truncated frame: need {needed} bytes, have {available}")]
    Truncated { needed: usize, available: usize },
    #[error("message {kind:#04x} has length {got}, expected {expected}")]
    LengthMismatch { kind: u8, expected: usize, got: usize },
    #[error("empty message")]
    Empty,
    #[error("invalid buy/sell indicator {0:#04x}")]
    InvalidSide(u8),
    #[error("value not representable in ITCH field")]
    Unrepresentable,
}

#[inline(always)]
fn be_u16(b: &[u8], o: usize) -> u16 {
    u16::from_be_bytes([b[o], b[o + 1]])
}

#[inline(always)]
fn be_u32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline(always)]
fn be_u48(b: &[u8], o: usize) -> u64 {
    u64::from_be_bytes([0, 0, b[o], b[o + 1], b[o + 2], b[o + 3], b[o + 4], b[o + 5]])
}

#[inline(always)]
fn be_u64(b: &[u8], o: usize) -> u64 {
    u64::from_be_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

#[inline(always)]
fn side(b: u8) -> Result<Side, ItchError> {
    match b {
        b'B' => Ok(Side::Bid),
        b'S' => Ok(Side::Ask),
        other => Err(ItchError::InvalidSide(other)),
    }
}

#[inline(always)]
fn price(b: &[u8], o: usize) -> Price {
    Price::from_raw(i64::from(be_u32(b, o)) * ITCH_PRICE_TO_RAW)
}

#[inline(always)]
fn shares(b: &[u8], o: usize) -> Qty {
    Qty::from_raw(i64::from(be_u32(b, o)) * SCALE)
}

#[inline(always)]
fn expect_len(msg: &[u8], kind: u8, expected: usize) -> Result<(), ItchError> {
    if msg.len() == expected {
        Ok(())
    } else {
        Err(ItchError::LengthMismatch {
            kind,
            expected,
            got: msg.len(),
        })
    }
}

/// Decodes one unframed message. Returns `Ok(None)` for well-formed message types that do
/// not affect the order book (system events, directory, imbalance, …).
///
/// `P` carries the side of the resting non-displayed order; the returned
/// [`EventBody::Trade`] carries the aggressor, i.e. the opposite side.
#[inline]
pub fn decode_message(msg: &[u8]) -> Result<Option<(Timestamp, EventBody)>, ItchError> {
    let Some(&kind) = msg.first() else {
        return Err(ItchError::Empty);
    };
    let body = match kind {
        MSG_ADD | MSG_ADD_MPID => {
            expect_len(msg, kind, if kind == MSG_ADD { LEN_ADD } else { LEN_ADD_MPID })?;
            EventBody::Add {
                order_id: OrderId(be_u64(msg, 11)),
                side: side(msg[19])?,
                qty: shares(msg, 20),
                price: price(msg, 32),
            }
        }
        MSG_EXECUTED => {
            expect_len(msg, kind, LEN_EXECUTED)?;
            EventBody::Execute {
                order_id: OrderId(be_u64(msg, 11)),
                qty: shares(msg, 19),
                match_id: be_u64(msg, 23),
            }
        }
        MSG_EXECUTED_WITH_PRICE => {
            expect_len(msg, kind, LEN_EXECUTED_WITH_PRICE)?;
            EventBody::Execute {
                order_id: OrderId(be_u64(msg, 11)),
                qty: shares(msg, 19),
                match_id: be_u64(msg, 23),
            }
        }
        MSG_CANCEL => {
            expect_len(msg, kind, LEN_CANCEL)?;
            EventBody::Cancel {
                order_id: OrderId(be_u64(msg, 11)),
                qty: shares(msg, 19),
            }
        }
        MSG_DELETE => {
            expect_len(msg, kind, LEN_DELETE)?;
            EventBody::Delete {
                order_id: OrderId(be_u64(msg, 11)),
            }
        }
        MSG_REPLACE => {
            expect_len(msg, kind, LEN_REPLACE)?;
            EventBody::Replace {
                order_id: OrderId(be_u64(msg, 11)),
                new_order_id: OrderId(be_u64(msg, 19)),
                qty: shares(msg, 27),
                price: price(msg, 31),
            }
        }
        MSG_TRADE => {
            expect_len(msg, kind, LEN_TRADE)?;
            EventBody::Trade {
                side: side(msg[19])?.opposite(),
                qty: shares(msg, 20),
                price: price(msg, 32),
                match_id: be_u64(msg, 36),
            }
        }
        _ => return Ok(None),
    };
    Ok(Some((Timestamp(be_u48(msg, 5)), body)))
}

/// Decodes the length-prefixed frame at the start of `buf`.
/// Returns the number of bytes consumed and the decoded event, if any.
#[inline]
pub fn decode_frame(buf: &[u8]) -> Result<(usize, Option<(Timestamp, EventBody)>), ItchError> {
    if buf.len() < 2 {
        return Err(ItchError::Truncated {
            needed: 2,
            available: buf.len(),
        });
    }
    let len = usize::from(be_u16(buf, 0));
    let end = 2 + len;
    if buf.len() < end {
        return Err(ItchError::Truncated {
            needed: end,
            available: buf.len(),
        });
    }
    Ok((end, decode_message(&buf[2..end])?))
}

/// Iterator over frames of a contiguous ITCH buffer.
#[derive(Debug, Clone)]
pub struct FrameCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> FrameCursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.buf.len()
    }

    #[inline]
    pub fn next_event(&mut self) -> Option<Result<(Timestamp, EventBody), ItchError>> {
        while self.pos < self.buf.len() {
            match decode_frame(&self.buf[self.pos..]) {
                Ok((n, decoded)) => {
                    self.pos += n;
                    if let Some(ev) = decoded {
                        return Some(Ok(ev));
                    }
                }
                Err(e) => {
                    self.pos = self.buf.len();
                    return Some(Err(e));
                }
            }
        }
        None
    }
}

/// Encoder used by the simulator, tests and benchmarks.
#[derive(Debug, Clone)]
pub struct ItchEncoder {
    pub stock_locate: u16,
    pub stock: [u8; 8],
    tracking: u16,
}

impl ItchEncoder {
    pub fn new(stock_locate: u16, symbol: &str) -> Self {
        let mut stock = [b' '; 8];
        for (dst, src) in stock.iter_mut().zip(symbol.bytes()) {
            *dst = src;
        }
        Self {
            stock_locate,
            stock,
            tracking: 0,
        }
    }

    fn header(&mut self, out: &mut Vec<u8>, kind: u8, len: usize, ts: Timestamp) -> Result<(), ItchError> {
        if ts.nanos() > MAX_TIMESTAMP {
            return Err(ItchError::Unrepresentable);
        }
        out.extend_from_slice(&(len as u16).to_be_bytes());
        out.push(kind);
        out.extend_from_slice(&self.stock_locate.to_be_bytes());
        out.extend_from_slice(&self.tracking.to_be_bytes());
        out.extend_from_slice(&ts.nanos().to_be_bytes()[2..]);
        self.tracking = self.tracking.wrapping_add(1);
        Ok(())
    }

    fn side_byte(side: Side) -> u8 {
        match side {
            Side::Bid => b'B',
            Side::Ask => b'S',
        }
    }

    fn price_field(p: Price) -> Result<[u8; 4], ItchError> {
        if p.raw() % ITCH_PRICE_TO_RAW != 0 {
            return Err(ItchError::Unrepresentable);
        }
        u32::try_from(p.raw() / ITCH_PRICE_TO_RAW)
            .map(u32::to_be_bytes)
            .map_err(|_| ItchError::Unrepresentable)
    }

    fn shares_field(q: Qty) -> Result<[u8; 4], ItchError> {
        if q.raw() % SCALE != 0 {
            return Err(ItchError::Unrepresentable);
        }
        u32::try_from(q.raw() / SCALE)
            .map(u32::to_be_bytes)
            .map_err(|_| ItchError::Unrepresentable)
    }

    /// Appends one framed message. Validates every field before writing, so on `Err` the
    /// buffer is unchanged.
    pub fn encode(&mut self, out: &mut Vec<u8>, ts: Timestamp, body: &EventBody) -> Result<(), ItchError> {
        match *body {
            EventBody::Add {
                order_id,
                side,
                price,
                qty,
            } => {
                let (shares, px) = (Self::shares_field(qty)?, Self::price_field(price)?);
                self.header(out, MSG_ADD, LEN_ADD, ts)?;
                out.extend_from_slice(&order_id.0.to_be_bytes());
                out.push(Self::side_byte(side));
                out.extend_from_slice(&shares);
                out.extend_from_slice(&self.stock);
                out.extend_from_slice(&px);
            }
            EventBody::Execute {
                order_id,
                qty,
                match_id,
            } => {
                let shares = Self::shares_field(qty)?;
                self.header(out, MSG_EXECUTED, LEN_EXECUTED, ts)?;
                out.extend_from_slice(&order_id.0.to_be_bytes());
                out.extend_from_slice(&shares);
                out.extend_from_slice(&match_id.to_be_bytes());
            }
            EventBody::Cancel { order_id, qty } => {
                let shares = Self::shares_field(qty)?;
                self.header(out, MSG_CANCEL, LEN_CANCEL, ts)?;
                out.extend_from_slice(&order_id.0.to_be_bytes());
                out.extend_from_slice(&shares);
            }
            EventBody::Delete { order_id } => {
                self.header(out, MSG_DELETE, LEN_DELETE, ts)?;
                out.extend_from_slice(&order_id.0.to_be_bytes());
            }
            EventBody::Replace {
                order_id,
                new_order_id,
                price,
                qty,
            } => {
                let (shares, px) = (Self::shares_field(qty)?, Self::price_field(price)?);
                self.header(out, MSG_REPLACE, LEN_REPLACE, ts)?;
                out.extend_from_slice(&order_id.0.to_be_bytes());
                out.extend_from_slice(&new_order_id.0.to_be_bytes());
                out.extend_from_slice(&shares);
                out.extend_from_slice(&px);
            }
            EventBody::Trade {
                side,
                price,
                qty,
                match_id,
            } => {
                let (shares, px) = (Self::shares_field(qty)?, Self::price_field(price)?);
                self.header(out, MSG_TRADE, LEN_TRADE, ts)?;
                out.extend_from_slice(&0u64.to_be_bytes());
                out.push(Self::side_byte(side.opposite()));
                out.extend_from_slice(&shares);
                out.extend_from_slice(&self.stock);
                out.extend_from_slice(&px);
                out.extend_from_slice(&match_id.to_be_bytes());
            }
        }
        Ok(())
    }

    pub fn encode_system_event(&mut self, out: &mut Vec<u8>, ts: Timestamp, code: u8) -> Result<(), ItchError> {
        self.header(out, MSG_SYSTEM_EVENT, 12, ts)?;
        out.push(code);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn px(v: f64) -> Price {
        Price::from_f64(v).unwrap()
    }

    #[test]
    fn layouts_match_spec_lengths() {
        let mut enc = ItchEncoder::new(7, "AAPL");
        let ts = Timestamp(34_200_000_000_123);
        let cases = [
            (
                EventBody::Add {
                    order_id: OrderId(1),
                    side: Side::Bid,
                    price: px(189.12),
                    qty: Qty::from_units(100).unwrap(),
                },
                LEN_ADD,
            ),
            (
                EventBody::Execute {
                    order_id: OrderId(1),
                    qty: Qty::from_units(40).unwrap(),
                    match_id: 9,
                },
                LEN_EXECUTED,
            ),
            (
                EventBody::Cancel {
                    order_id: OrderId(1),
                    qty: Qty::from_units(10).unwrap(),
                },
                LEN_CANCEL,
            ),
            (EventBody::Delete { order_id: OrderId(1) }, LEN_DELETE),
            (
                EventBody::Replace {
                    order_id: OrderId(1),
                    new_order_id: OrderId(2),
                    price: px(189.13),
                    qty: Qty::from_units(5).unwrap(),
                },
                LEN_REPLACE,
            ),
            (
                EventBody::Trade {
                    side: Side::Ask,
                    price: px(189.10),
                    qty: Qty::from_units(3).unwrap(),
                    match_id: 11,
                },
                LEN_TRADE,
            ),
        ];
        for (body, len) in cases {
            let mut buf = Vec::new();
            enc.encode(&mut buf, ts, &body).unwrap();
            assert_eq!(buf.len(), len + 2);
            assert_eq!(usize::from(u16::from_be_bytes([buf[0], buf[1]])), len);
            let (consumed, decoded) = decode_frame(&buf).unwrap();
            assert_eq!(consumed, buf.len());
            assert_eq!(decoded, Some((ts, body)));
        }
    }

    #[test]
    fn decodes_reference_add_order_bytes() {
        let msg: [u8; 36] = [
            b'A', 0x00, 0x01, 0x00, 0x02, 0x1F, 0x1A, 0x6E, 0x2C, 0x40, 0x00, 0, 0, 0, 0, 0, 0, 0x30, 0x39, b'S', 0x00,
            0x00, 0x01, 0x2C, b'M', b'S', b'F', b'T', b' ', b' ', b' ', b' ', 0x00, 0x05, 0x7E, 0x40,
        ];
        let (ts, body) = decode_message(&msg).unwrap().unwrap();
        assert_eq!(ts, Timestamp(0x1F1A_6E2C_4000));
        assert_eq!(
            body,
            EventBody::Add {
                order_id: OrderId(12_345),
                side: Side::Ask,
                qty: Qty::from_units(300).unwrap(),
                price: px(36.0)
            }
        );
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(
            decode_frame(&[0x00]),
            Err(ItchError::Truncated {
                needed: 2,
                available: 1
            })
        );
        assert_eq!(
            decode_frame(&[0x00, 0x05, b'D']),
            Err(ItchError::Truncated {
                needed: 7,
                available: 3
            })
        );
        assert_eq!(decode_message(&[]), Err(ItchError::Empty));
        assert_eq!(
            decode_message(&[b'D'; 18]),
            Err(ItchError::LengthMismatch {
                kind: b'D',
                expected: 19,
                got: 18
            })
        );
        let mut add = [0u8; 36];
        add[0] = b'A';
        add[19] = b'Z';
        assert_eq!(decode_message(&add), Err(ItchError::InvalidSide(b'Z')));
        assert_eq!(decode_message(&[b'R'; 39]), Ok(None));
    }

    #[test]
    fn encoder_rejects_unrepresentable_fields() {
        let mut enc = ItchEncoder::new(1, "X");
        let mut buf = Vec::new();
        let sub_tick = EventBody::Add {
            order_id: OrderId(1),
            side: Side::Bid,
            price: Price::from_raw(1),
            qty: Qty::from_units(1).unwrap(),
        };
        assert_eq!(
            enc.encode(&mut buf, Timestamp(0), &sub_tick),
            Err(ItchError::Unrepresentable)
        );
        assert!(buf.is_empty());
        let late = EventBody::Delete { order_id: OrderId(1) };
        assert_eq!(
            enc.encode(&mut buf, Timestamp(1 << 48), &late),
            Err(ItchError::Unrepresentable)
        );
    }

    #[test]
    fn cursor_skips_non_book_messages() {
        let mut enc = ItchEncoder::new(1, "X");
        let mut buf = Vec::new();
        enc.encode_system_event(&mut buf, Timestamp(1), b'O').unwrap();
        enc.encode(&mut buf, Timestamp(2), &EventBody::Delete { order_id: OrderId(5) })
            .unwrap();
        let mut cur = FrameCursor::new(&buf);
        assert_eq!(
            cur.next_event(),
            Some(Ok((Timestamp(2), EventBody::Delete { order_id: OrderId(5) })))
        );
        assert_eq!(cur.next_event(), None);
        assert!(cur.is_exhausted());
    }

    proptest! {
        #[test]
        fn decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..64)) {
            let _ = decode_frame(&bytes);
            let _ = decode_message(&bytes);
        }

        #[test]
        fn add_round_trips(id in 0u64..u64::MAX, bid in any::<bool>(), shares in 1u32.., px4 in 1u32..) {
            let body = EventBody::Add {
                order_id: OrderId(id),
                side: if bid { Side::Bid } else { Side::Ask },
                price: Price::from_raw(i64::from(px4) * ITCH_PRICE_TO_RAW),
                qty: Qty::from_raw(i64::from(shares) * SCALE),
            };
            let mut buf = Vec::new();
            ItchEncoder::new(3, "ZZZZ").encode(&mut buf, Timestamp(123_456_789), &body).unwrap();
            prop_assert_eq!(decode_frame(&buf).unwrap(), (buf.len(), Some((Timestamp(123_456_789), body))));
        }
    }
}
