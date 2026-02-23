// file: src/sampler.rs
// description: Implements greedy and nucleus sampling over logits.
// author: cipher-rc5

use anyhow::Result;

pub fn sample_next_token(logits: &[f32], temperature: f32, top_p: f32) -> Result<u32> {
    anyhow::ensure!(!logits.is_empty(), "empty logits");
    if temperature == 0.0 {
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        return Ok(idx as u32);
    }

    let mut scaled = logits.to_vec();
    for l in &mut scaled {
        *l /= temperature.max(1e-6);
    }

    let max_l = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scaled.iter().map(|x| (*x - max_l).exp()).collect();
    let z: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= z.max(1e-12);
    }

    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept = Vec::new();
    let mut cum = 0.0;
    for idx in order {
        kept.push(idx);
        cum += probs[idx];
        if cum >= top_p {
            break;
        }
    }

    let r = random_f32();
    let mut c = 0.0;
    for idx in kept {
        c += probs[idx];
        if r <= c {
            return Ok(idx as u32);
        }
    }
    Ok(0)
}

fn random_f32() -> f32 {
    use std::cell::RefCell;
    thread_local! {
      static RNG_STATE: RefCell<u64> = const { RefCell::new(0x1234_5678_9abc_def0) };
    }
    RNG_STATE.with(|state| {
        let mut s = state.borrow_mut();
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s as f64) / (u64::MAX as f64)) as f32
    })
}
