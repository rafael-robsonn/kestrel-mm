use core_types::Timestamp;

use crate::AlphaError;

type M3 = [[f64; 3]; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EkfConfig {
    /// Minimum exchange-time spacing between filter steps. Observations arriving faster
    /// are ignored, which keeps the squared-return measurement above the tick-noise floor.
    pub sample_interval_ns: u64,
    /// Mean-reversion time of the drift state, seconds.
    pub drift_tau_s: f64,
    /// Drift diffusion, ticks²/s³.
    pub drift_noise: f64,
    /// Long-run mean of `h = ln σ²` with `σ²` in ticks²/s.
    pub logvar_mean: f64,
    /// Mean-reversion time of `h`, seconds.
    pub logvar_tau_s: f64,
    /// Diffusion of `h`, per second.
    pub logvar_noise: f64,
    /// Variance of the micro-price around the fair value, ticks².
    pub obs_noise: f64,
    pub logvar_min: f64,
    pub logvar_max: f64,
    /// Squared-return innovations are clipped to `[-ŷ, clip · ŷ]`, `ŷ` the prediction.
    pub innovation_clip: f64,
}

impl Default for EkfConfig {
    fn default() -> Self {
        Self {
            sample_interval_ns: 100_000_000,
            drift_tau_s: 2.0,
            drift_noise: 4.0,
            logvar_mean: 2.0,
            logvar_tau_s: 60.0,
            logvar_noise: 0.05,
            obs_noise: 0.02,
            logvar_min: -8.0,
            logvar_max: 12.0,
            innovation_clip: 8.0,
        }
    }
}

impl EkfConfig {
    fn validate(&self) -> Result<(), AlphaError> {
        let positive = [
            self.drift_tau_s,
            self.drift_noise,
            self.logvar_tau_s,
            self.logvar_noise,
            self.obs_noise,
            self.innovation_clip,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(AlphaError::InvalidConfig(
                "EKF noise and time constants must be finite and positive",
            ));
        }
        let bounds = [self.logvar_min, self.logvar_max, self.logvar_mean];
        if bounds.iter().any(|v| !v.is_finite()) || self.logvar_min >= self.logvar_max {
            return Err(AlphaError::InvalidConfig("EKF log-variance bounds invalid"));
        }
        if self.sample_interval_ns == 0 {
            return Err(AlphaError::InvalidConfig("EKF sample interval must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct EkfEstimate {
    pub ts: Timestamp,
    /// Filtered fair value, ticks.
    pub fair_value: f64,
    /// Drift, ticks/s.
    pub drift: f64,
    /// Diffusion `σ = exp(h/2)`, ticks/√s.
    pub sigma: f64,
    /// Posterior variance of the fair value, ticks².
    pub fair_value_var: f64,
}

/// Extended Kalman filter on `x = [m, μ, h]`: fair value, drift, log instantaneous variance.
///
/// Transition over `Δt` (linear, with state-dependent noise):
///
/// `m' = m + μΔt`, `μ' = e^{-Δt/τ_μ} μ`, `h' = h̄ + e^{-Δt/τ_h}(h - h̄)`
/// `Q = diag(e^h Δt, q_μ Δt, q_h Δt)`
///
/// Measurements: micro-price `z` and squared return `r² = (z - z_prev)²`:
///
/// `h(x) = [m, e^h Δt + μ²Δt² + 2R]`, `R = diag(R, 2·ĥ₂²)` (Gaussian `Var(r²) = 2E[r²]²`)
///
/// The second row is the only nonlinearity; its Jacobian is `[0, 2μΔt², e^h Δt]`.
/// Covariance uses the Joseph form and is re-symmetrised every step. All arithmetic is on
/// stack arrays: one step is O(1) with three `exp` calls.
#[derive(Clone, Debug)]
pub struct FairValueEkf {
    cfg: EkfConfig,
    x: [f64; 3],
    p: M3,
    last_z: f64,
    last_ts: Timestamp,
    initialized: bool,
    est: EkfEstimate,
}

#[inline(always)]
fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0].mul_add(b[0][j], a[i][1].mul_add(b[1][j], a[i][2] * b[2][j]));
        }
    }
    c
}

#[inline(always)]
fn transpose(a: &M3) -> M3 {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

impl FairValueEkf {
    pub fn new(cfg: EkfConfig) -> Result<Self, AlphaError> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            x: [0.0; 3],
            p: [[0.0; 3]; 3],
            last_z: 0.0,
            last_ts: Timestamp(0),
            initialized: false,
            est: EkfEstimate::default(),
        })
    }

    pub fn reset(&mut self) {
        self.initialized = false;
        self.est = EkfEstimate::default();
    }

    pub fn estimate(&self) -> &EkfEstimate {
        &self.est
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn init(&mut self, z: f64, ts: Timestamp) {
        self.x = [z, 0.0, self.cfg.logvar_mean];
        self.p = [
            [self.cfg.obs_noise, 0.0, 0.0],
            [0.0, self.cfg.drift_noise, 0.0],
            [0.0, 0.0, 1.0],
        ];
        self.last_z = z;
        self.last_ts = ts;
        self.initialized = true;
        self.publish(ts);
    }

    fn publish(&mut self, ts: Timestamp) {
        self.est = EkfEstimate {
            ts,
            fair_value: self.x[0],
            drift: self.x[1],
            sigma: (0.5 * self.x[2]).exp(),
            fair_value_var: self.p[0][0],
        };
    }

    /// Feeds one micro-price observation `z` (ticks) at exchange time `ts`.
    /// Returns `Ok(true)` if a predict/update step ran, `Ok(false)` if the observation fell
    /// inside the sampling interval and was skipped.
    ///
    /// On numerical divergence the filter re-initialises at `z` and returns `Err`.
    #[inline]
    pub fn observe(&mut self, z: f64, ts: Timestamp) -> Result<bool, AlphaError> {
        if !z.is_finite() {
            return Err(AlphaError::NonFiniteInput);
        }
        if !self.initialized {
            self.init(z, ts);
            return Ok(true);
        }
        let elapsed = ts.nanos().saturating_sub(self.last_ts.nanos());
        if elapsed < self.cfg.sample_interval_ns {
            return Ok(false);
        }
        let dt = (elapsed as f64 * 1e-9).min(10.0);
        let cfg = &self.cfg;

        let a = (-dt / cfg.drift_tau_s).exp();
        let b = (-dt / cfg.logvar_tau_s).exp();
        let [m, mu, h] = self.x;
        let xp = [m + mu * dt, a * mu, cfg.logvar_mean + b * (h - cfg.logvar_mean)];
        let f: M3 = [[1.0, dt, 0.0], [0.0, a, 0.0], [0.0, 0.0, b]];
        let mut pp = mat_mul(&mat_mul(&f, &self.p), &transpose(&f));
        pp[0][0] += h.exp() * dt;
        pp[1][1] += cfg.drift_noise * dt;
        pp[2][2] += cfg.logvar_noise * dt;

        let var_dt = xp[2].exp() * dt;
        let drift_dt = xp[1] * dt;
        let pred_r2 = var_dt + drift_dt * drift_dt + 2.0 * cfg.obs_noise;
        let r = z - self.last_z;
        let y = [
            z - xp[0],
            (r * r - pred_r2).clamp(-pred_r2, cfg.innovation_clip * pred_r2),
        ];
        let h21 = 2.0 * drift_dt * dt;
        let h22 = var_dt;
        let rn = [cfg.obs_noise, 2.0 * pred_r2 * pred_r2];

        let mut pht = [[0.0; 2]; 3];
        for i in 0..3 {
            pht[i][0] = pp[i][0];
            pht[i][1] = pp[i][1].mul_add(h21, pp[i][2] * h22);
        }
        let s00 = pht[0][0] + rn[0];
        let s01 = pht[0][1];
        let s11 = h21.mul_add(pht[1][1], h22 * pht[2][1]) + rn[1];
        let det = s00 * s11 - s01 * s01;
        if !det.is_finite() || det <= f64::MIN_POSITIVE {
            self.init(z, ts);
            return Err(AlphaError::SingularInnovation);
        }
        let inv = [[s11 / det, -s01 / det], [-s01 / det, s00 / det]];
        let mut k = [[0.0; 2]; 3];
        for i in 0..3 {
            k[i][0] = pht[i][0].mul_add(inv[0][0], pht[i][1] * inv[1][0]);
            k[i][1] = pht[i][0].mul_add(inv[0][1], pht[i][1] * inv[1][1]);
        }

        let mut xn = xp;
        for i in 0..3 {
            xn[i] += k[i][0].mul_add(y[0], k[i][1] * y[1]);
        }
        xn[2] = xn[2].clamp(cfg.logvar_min, cfg.logvar_max);

        let hm = [[1.0, 0.0, 0.0], [0.0, h21, h22]];
        let mut ikh: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                ikh[i][j] -= k[i][0].mul_add(hm[0][j], k[i][1] * hm[1][j]);
            }
        }
        let mut pn = mat_mul(&mat_mul(&ikh, &pp), &transpose(&ikh));
        for i in 0..3 {
            for j in 0..3 {
                pn[i][j] += k[i][0] * rn[0] * k[j][0] + k[i][1] * rn[1] * k[j][1];
            }
        }
        for i in 0..3 {
            for j in (i + 1)..3 {
                let s = 0.5 * (pn[i][j] + pn[j][i]);
                pn[i][j] = s;
                pn[j][i] = s;
            }
        }

        let finite = xn.iter().all(|v| v.is_finite()) && pn.iter().flatten().all(|v| v.is_finite());
        if !finite || pn[0][0] <= 0.0 || pn[1][1] <= 0.0 || pn[2][2] <= 0.0 {
            self.init(z, ts);
            return Err(AlphaError::Diverged);
        }
        self.x = xn;
        self.p = pn;
        self.last_z = z;
        self.last_ts = ts;
        self.publish(ts);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    fn every_10ms() -> EkfConfig {
        EkfConfig {
            sample_interval_ns: 10_000_000,
            ..EkfConfig::default()
        }
    }

    fn run(sigma: f64, drift: f64, steps: usize, seed: u64) -> EkfEstimate {
        let cfg = EkfConfig {
            obs_noise: 0.01,
            ..every_10ms()
        };
        let mut ekf = FairValueEkf::new(cfg).unwrap();
        let mut rng = Rng(seed);
        let dt = 0.01;
        let mut m = 10_000.0;
        for i in 0..steps {
            m += drift * dt + sigma * dt.sqrt() * rng.normal();
            let z = m + 0.1 * rng.normal();
            ekf.observe(z, Timestamp(i as u64 * 10_000_000)).unwrap();
        }
        *ekf.estimate()
    }

    #[test]
    fn recovers_diffusion_scale() {
        for (sigma, seed) in [(2.0, 7), (8.0, 11), (25.0, 13)] {
            let est = run(sigma, 0.0, 20_000, seed);
            let ratio = est.sigma / sigma;
            assert!((0.75..1.33).contains(&ratio), "sigma {sigma}: estimated {}", est.sigma);
        }
    }

    #[test]
    fn tracks_persistent_drift() {
        let est = run(2.0, 40.0, 3_000, 17);
        assert!(est.drift > 10.0, "drift {}", est.drift);
    }

    #[test]
    fn skips_sub_interval_and_rejects_nan() {
        let mut ekf = FairValueEkf::new(every_10ms()).unwrap();
        assert_eq!(ekf.observe(100.0, Timestamp(0)), Ok(true));
        assert_eq!(ekf.observe(100.0, Timestamp(1_000)), Ok(false));
        assert_eq!(
            ekf.observe(f64::NAN, Timestamp(20_000_000)),
            Err(AlphaError::NonFiniteInput)
        );
        assert_eq!(ekf.observe(101.0, Timestamp(20_000_000)), Ok(true));
        assert!(ekf.estimate().fair_value > 100.0 && ekf.estimate().fair_value <= 101.0);
    }

    #[test]
    fn survives_price_jump() {
        let mut ekf = FairValueEkf::new(every_10ms()).unwrap();
        for i in 0..500u64 {
            let z = if i < 250 { 10_000.0 } else { 12_000.0 };
            ekf.observe(z, Timestamp(i * 10_000_000)).unwrap();
        }
        let e = ekf.estimate();
        assert!((e.fair_value - 12_000.0).abs() < 1.0);
        assert!(e.sigma.is_finite() && e.fair_value_var > 0.0);
    }

    #[test]
    fn rejects_bad_config() {
        let cfg = EkfConfig {
            obs_noise: 0.0,
            ..EkfConfig::default()
        };
        assert!(FairValueEkf::new(cfg).is_err());
    }
}
