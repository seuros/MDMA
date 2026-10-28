use super::*;

#[test]
fn greedy_masks_controls_and_uses_lowest_index_ties() {
    let controls = Controls { think: 0, think_end: 1, entry: 2, exit: 3, eos: 4, im_end: None };
    assert_eq!(greedy(&[1.0; 6], controls, false).unwrap(), 0);
    assert_eq!(greedy(&[1.0; 6], controls, true).unwrap(), 1);
    assert_eq!(greedy(&[0.0, 0.0, 5.0, 10.0, 1.0], controls, false).unwrap(), 4);
    assert_eq!(greedy(&[0.0, 0.0, 5.0, 10.0, 1.0], controls, true).unwrap(), 2);
    for logits in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![f32::NEG_INFINITY]] {
        assert!(greedy(&logits, controls, false).is_err());
    }
}
