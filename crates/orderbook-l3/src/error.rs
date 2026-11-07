use core_types::{OrderId, Price, Qty};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BookError {
    #[error("invalid book configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("order id is reserved")]
    ReservedOrderId,
    #[error("duplicate order id {0:?}")]
    DuplicateOrderId(OrderId),
    #[error("unknown order id {0:?}")]
    UnknownOrder(OrderId),
    #[error("quantity must be strictly positive, got {0}")]
    InvalidQuantity(Qty),
    #[error("price {0} outside configured ladder")]
    PriceOutOfRange(Price),
    #[error("price {0} not aligned to tick")]
    TickMisaligned(Price),
    #[error("order arena exhausted")]
    CapacityExhausted,
    #[error("aggregate quantity overflow")]
    QuantityOverflow,
    #[error("execution of {requested} exceeds open quantity {open} on {order_id:?}")]
    ExecutionExceedsOpen {
        order_id: OrderId,
        open: Qty,
        requested: Qty,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("book invariant violated: {0}")]
pub struct AuditError(pub &'static str);
