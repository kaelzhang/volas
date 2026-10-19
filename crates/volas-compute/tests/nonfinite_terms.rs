//! Term/window owners also cover kernels whose NaN propagation was implicit IEEE math.
use volas_compute::raw::indicators as ind;

fn outputs(o: &[f64], h: &[f64], l: &[f64], c: &[f64], v: &[f64]) -> Vec<Vec<f64>> {
    let mut out = vec![
        ind::alma(c, 5, 0.85, 6.0), ind::swma(c), ind::hma(c, 1, 0),
        ind::avgprice(o, h, l, c), ind::medprice(h, l), ind::typprice(h, l, c),
        ind::wclprice(h, l, c), ind::bop(o, h, l, c), ind::change(c, 2),
        ind::iii(h, l, c, v), ind::emv(h, l, v, 5, 0), ind::kst(c, 0),
        ind::coppock(c, 5, 14, 11, 0), ind::wvad(o, h, l, c, v, 5, 0),
        ind::ichimoku(h, l, c, 9, 26, 52, ind::IchimokuLine::Chikou),
    ];
    for values in [ind::increase(c, 2, 1), ind::style(ind::Style::Bullish, o, c), ind::style(ind::Style::Bearish, o, c)] {
        out.push(values.into_iter().map(|v| v.map_or(f64::NAN, |b| if b { 1.0 } else { 0.0 })).collect());
    }
    for line in [ind::CdpLine::Cdp, ind::CdpLine::Ah, ind::CdpLine::Nh, ind::CdpLine::Nl, ind::CdpLine::Al] {
        out.push(ind::cdp(h, l, c, line));
    }
    for line in [ind::PivotLine::Pp, ind::PivotLine::R1, ind::PivotLine::R2, ind::PivotLine::R3,
        ind::PivotLine::S1, ind::PivotLine::S2, ind::PivotLine::S3] {
        out.push(ind::pivot_points(h, l, c, line));
    }
    for line in [ind::MikeLine::WeakR, ind::MikeLine::MidR, ind::MikeLine::StrongR,
        ind::MikeLine::WeakS, ind::MikeLine::MidS, ind::MikeLine::StrongS] {
        out.push(ind::mike(h, l, c, 5, line));
    }
    out
}

#[test]
fn every_read_term_treats_either_infinity_as_nan() {
    let c: Vec<_> = (0..80).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
    let original = [c.iter().map(|x| x - 1.0).collect(), c.iter().map(|x| x + 2.0).collect(),
        c.iter().map(|x| x - 2.0).collect(), c, vec![1000.0; 80]];
    for slot in 0..5 {
        for lane in [0, 1, 31, 63, 79] {
            for bad in [f64::INFINITY, f64::NEG_INFINITY] {
                let mut inf = original.clone();
                let mut nan = original.clone();
                inf[slot][lane] = bad;
                nan[slot][lane] = f64::NAN;
                let a = outputs(&inf[0], &inf[1], &inf[2], &inf[3], &inf[4]);
                let b = outputs(&nan[0], &nan[1], &nan[2], &nan[3], &nan[4]);
                for (kernel, (a, b)) in a.iter().zip(&b).enumerate() {
                    for (row, (&x, &y)) in a.iter().zip(b).enumerate() {
                        assert!(x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan(),
                            "kernel={kernel}, slot={slot}, lane={lane}, row={row}: {x} vs {y}");
                    }
                }
            }
        }
    }
    // An overflow of the kernel's own arithmetic from finite inputs keeps IEEE.
    assert!(ind::avgprice(&[f64::MAX], &[f64::MAX], &[f64::MAX], &[f64::MAX])[0].is_infinite());
}
