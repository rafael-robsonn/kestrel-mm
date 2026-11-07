#![allow(clippy::unwrap_used, clippy::expect_used, unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use core_types::MarketEvent;
use engine_runtime::clock::ticks;
use engine_runtime::itch::FrameCursor;
use engine_runtime::ring::spsc;
use engine_runtime::sim::{MarketSimulator, SimConfig, encode_itch};
use engine_runtime::{EngineConfig, StrategyEngine};

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
}

fn note_allocation() {
    if TRACKING.try_with(Cell::get).unwrap_or(false) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: forwards every call unchanged to the system allocator; the counter has no effect
// on the returned memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_allocation();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn tracked<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    TRACKING.with(|t| t.set(true));
    let r = f();
    TRACKING.with(|t| t.set(false));
    (r, ALLOCATIONS.load(Ordering::Relaxed) - before)
}

#[test]
fn hot_path_performs_no_heap_allocation() {
    let cfg = SimConfig {
        seed: 7,
        ..SimConfig::default()
    };
    let mut sim = MarketSimulator::new(cfg).unwrap();
    let events = sim.generate(250_000);
    let feed = encode_itch(&events, "SIMX").unwrap();
    let mut engine =
        StrategyEngine::new(EngineConfig::for_instrument(cfg.start_price, cfg.tick, cfg.lot).unwrap()).unwrap();
    let (mut tx, mut rx) = spsc::<MarketEvent>(1 << 10).unwrap();

    let (_, sanity) = tracked(|| std::hint::black_box(Vec::<u64>::with_capacity(8)));
    assert_eq!(sanity, 1, "allocator instrumentation is not active");

    let ((processed, quotes), allocations) = tracked(|| {
        let mut cursor = FrameCursor::new(&feed);
        let mut processed = 0u64;
        let mut quotes = 0u64;
        while let Some(Ok((exch_ts, body))) = cursor.next_event() {
            let ev = MarketEvent {
                seq: processed,
                exch_ts,
                recv_tsc: ticks(),
                body,
            };
            tx.push(ev).unwrap();
            let ev = rx.pop().unwrap();
            if let Ok(Some(q)) = engine.on_event(&ev) {
                quotes += 1;
                std::hint::black_box(q);
            }
            processed += 1;
        }
        (processed, quotes)
    });

    assert_eq!(processed as usize, events.len());
    assert!(quotes > 0);
    assert_eq!(
        allocations, 0,
        "hot path allocated {allocations} times over {processed} events"
    );

    let (_, after_reset) = tracked(|| engine.reset());
    assert_eq!(after_reset, 0, "reset must reuse existing buffers");
}
