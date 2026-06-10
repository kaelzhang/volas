    use super::*;
    use crate::indicators::test_support::*;

    /// SAR resume, fed the carried `[is_long, af, ep, sar, prev_high, prev_low]` of a full
    /// compute over the head, reproduces the tail of a full compute over the whole input —
    /// bit-for-bit. Iterating `from` over an oscillating OHLC series places reversals on
    /// BOTH sides of the cut, firing the long->short (731-737) and short->long (746-752)
    /// reversal arms plus both trend-continue arms.
    #[test]
    fn sar_resume_is_bit_identical_to_full() {
        let (high, low, _close) = ohlc(160);
        let (accel, max) = (0.02, 0.2);
        let full = sar(&high, &low, accel, max);
        for &from in &[2usize, 3, 10, 40, 80, 120, 159] {
            let st = sar_final_state(&high[..from], &low[..from], accel, max).unwrap();
            let (tail, _) = sar_resume(&high, &low, accel, max, from, &st).unwrap();
            assert_bits(&tail, &full[from..], "sar");
        }
        // The oscillating series must actually reverse in both directions for the resume to
        // have exercised both reversal arms — confirm the full SAR sign flips at least once
        // each way (SAR itself is unsigned, so detect reversals via the value jumps).
        let mut saw_up = false;
        let mut saw_down = false;
        for w in full.windows(2) {
            if w[1] > w[0] {
                saw_up = true;
            }
            if w[1] < w[0] {
                saw_down = true;
            }
        }
        assert!(saw_up && saw_down, "fixture must swing both ways");
    }

    /// SAREXT resume (signed output) reproduces the full tail bit-for-bit, with a non-zero
    /// `offset_on_reverse` so the reversal-offset nudges (997-998 / 1015-1016) fire too.
    #[test]
    fn sarext_resume_is_bit_identical_to_full() {
        let (high, low, _close) = ohlc(160);
        // Asymmetric long/short acceleration + a reversal offset.
        let (offset, ail, al, aml, ais, as_, ams) = (0.1, 0.02, 0.02, 0.2, 0.03, 0.03, 0.25);
        for &start in &[0.0_f64, 1.0, -1.0] {
            let full = sarext(&high, &low, start, offset, ail, al, aml, ais, as_, ams);
            for &from in &[2usize, 5, 30, 70, 110, 159] {
                let st = sarext_final_state(
                    &high[..from],
                    &low[..from],
                    start,
                    offset,
                    ail,
                    al,
                    aml,
                    ais,
                    as_,
                    ams,
                )
                .unwrap();
                let (tail, _) =
                    sarext_resume(&high, &low, offset, ail, al, aml, ais, as_, ams, from, &st)
                        .unwrap();
                assert_bits(&tail, &full[from..], "sarext");
            }
        }
    }

    /// SAREXT `*_final_state` bootstrap branches: a positive start forces an initial long at
    /// that level (900 / 904-905), a negative start an initial short at `|start|` (906-907),
    /// and a non-zero offset nudges the very first reversal (922 / 938).
    #[test]
    fn sarext_final_state_bootstrap_and_offset() {
        let (high, low, _close) = ohlc(80);
        let (al, aml, as_, ams) = (0.02, 0.2, 0.02, 0.2);
        // Exercise all three bootstrap arms of `*_final_state`: the SAR `-DM1` directional
        // bootstrap (start == 0, 902-903) and the forced long / short starts (start > 0 at
        // 904-905, start < 0 at 906-907). The first *computed* SAREXT value (`out[1]`) is
        // seeded directly from `sar`, so the three starts produce visibly different series.
        let bootstrap = |start: f64| -> (Vec<f64>, Vec<f64>) {
            let full = sarext(&high, &low, start, 0.0, al, al, aml, as_, as_, ams);
            let st =
                sarext_final_state(&high, &low, start, 0.0, al, al, aml, as_, as_, ams).unwrap();
            (full, st)
        };
        let (full_zero, _) = bootstrap(0.0);
        let (full_long, _) = bootstrap(5.0); // forced long: sar seeded from +start (904-905)
        let (full_short, _) = bootstrap(-5.0); // forced short: sar seeded from |start| (906-907)
                                               // The three distinct bootstrap arms seed distinct stops, so the first computed value
                                               // differs across all three (proving the forced-long and forced-short arms both ran,
                                               // not just the directional `start == 0` arm).
        assert_ne!(
            full_zero[1].to_bits(),
            full_long[1].to_bits(),
            "long bootstrap distinct"
        );
        assert_ne!(
            full_zero[1].to_bits(),
            full_short[1].to_bits(),
            "short bootstrap distinct"
        );
        assert_ne!(
            full_long[1].to_bits(),
            full_short[1].to_bits(),
            "long != short bootstrap"
        );

        // With offset != 0 the carried `sar` differs from the offset == 0 run (the reversal
        // nudge at 922/938 changed it). A long enough oscillating series has reversed by n-1.
        let s_no_off =
            sarext_final_state(&high, &low, 0.0, 0.0, al, al, aml, as_, as_, ams).unwrap();
        let s_off = sarext_final_state(&high, &low, 0.0, 0.1, al, al, aml, as_, as_, ams).unwrap();
        assert_ne!(
            s_no_off[4].to_bits(),
            s_off[4].to_bits(),
            "offset must move the stop"
        );
    }

    /// SAR / SAREXT warm-up guards: `*_final_state` declines for `n < 2` (no SAR value), and
    /// `*_resume` declines for `from < 2` (the bootstrap reads bars 0 and 1, never re-run).
    #[test]
    fn sar_guards_decline() {
        let one_h = [10.0];
        let one_l = [8.0];
        // n < 2 -> no state (654 / 888).
        assert!(sar_final_state(&one_h, &one_l, 0.02, 0.2).is_none());
        assert!(
            sarext_final_state(&one_h, &one_l, 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2)
                .is_none()
        );
        // from < 2 -> resume declines (714 / 973). State is unread on the None path.
        let st = vec![1.0, 0.02, 10.0, 8.0, 10.0, 8.0];
        let (h, l, _c) = ohlc(20);
        assert!(sar_resume(&h, &l, 0.02, 0.2, 1, &st).is_none());
        let st7 = vec![1.0, 0.02, 0.02, 10.0, 8.0, 10.0, 8.0];
        assert!(sarext_resume(&h, &l, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2, 1, &st7).is_none());
    }

    /// The remaining `*_final_state` / `*_resume` None guards in the MA-family helpers:
    /// `ema_seed_idx` period==0 (53), `kama_final_state` no-seed (500) + leading-NaN
    /// recursion (506), `kama_resume` underflow (545), and `atr_final_state` /
    /// `atr_resume` guards (1226 / 1253).
    #[test]
    fn ma_family_state_guards() {
        let data = series(60);
        // ema_seed_idx period == 0 -> ema_final_state returns None (trend.rs:53).
        assert!(ema_final_state(&data, 0).is_none());

        // kama_final_state: period == 0 / period + 1 > n -> None (trend.rs:500).
        assert!(kama_final_state(&data, 0).is_none());
        assert!(kama_final_state(&data[..5], 30).is_none());
        // kama_final_state leading-NaN prefix -> recurse on the finite tail (trend.rs:506).
        let mut nan_head = series(60);
        for x in nan_head.iter_mut().take(3) {
            *x = f64::NAN;
        }
        // A finite-tail KAMA state still computes; it must equal the state of the tail alone.
        let via_head = kama_final_state(&nan_head, 10).unwrap();
        let via_tail = kama_final_state(&nan_head[3..], 10).unwrap();
        assert_eq!(
            via_head[0].to_bits(),
            via_tail[0].to_bits(),
            "kama leading-NaN recursion"
        );
        assert_eq!(via_head[1].to_bits(), via_tail[1].to_bits());

        // kama_resume underflow: period == 0 / from <= period / from > n -> None (trend.rs:545).
        let st = kama_final_state(&data, 10).unwrap();
        assert!(kama_resume(&data, 0, 20, &st).is_none());
        assert!(kama_resume(&data, 10, 10, &st).is_none()); // from <= period
        assert!(kama_resume(&data, 10, 1000, &st).is_none()); // from > n

        // atr_final_state: period == 0 / period >= n -> None (trend.rs:1226).
        let (h, l, c) = ohlc(60);
        assert!(atr_final_state(&h, &l, &c, 0).is_none());
        assert!(atr_final_state(&h, &l, &c, 100).is_none());
        // atr_resume: from == 0 -> None (trend.rs:1253).
        let st = atr_final_state(&h, &l, &c, 14).unwrap();
        assert!(atr_resume(&h, &l, &c, 14, 0, &st).is_none());
    }
