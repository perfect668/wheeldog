"""Check pairing analysis cannot silently choose a favorable single direction."""
from h1_capability.analysis import pair_metrics, select_pairs
from h1_capability.model import DIRECTIONS


def test_primary_score_uses_six_directions_and_is_pair_symmetric():
    lookup={}
    for candidate in ['a','b']:
        for mobile in [False,True]:
            for direction in DIRECTIONS:
                displacement=.02 if candidate=='a' or direction!='left' else .08
                lookup[(candidate,.3,mobile,direction)]={
                    'achieved_displacement_m':displacement,'status':'VERIFIED_PATH_FOUND'}
    p=dict(a='a',b='b',max_relative_gap=.03,status='MATCHED')
    r=pair_metrics(p,lookup,[.3],.01,1.25)
    assert abs(r['primary_mean_six_direction_gap_m']-.01)<1e-12
    assert r['primary_practical_directions']==1
    swapped=pair_metrics(dict(p,a='b',b='a'),lookup,[.3],.01,1.25)
    assert swapped['primary_mean_six_direction_gap_m']==r['primary_mean_six_direction_gap_m']
    assert len(r['comparisons'])==12


def test_representative_pairs_are_disjoint_and_missing_path_not_zero():
    metrics=[dict(a='a',b='b',primary_mean_six_direction_gap_m=.04),
             dict(a='a',b='c',primary_mean_six_direction_gap_m=.03),
             dict(a='c',b='d',primary_mean_six_direction_gap_m=.02),
             dict(a='e',b='f',primary_mean_six_direction_gap_m=None)]
    selected=select_pairs(metrics,3)
    assert [(p['a'],p['b']) for p in selected]==[('a','b'),('c','d')]
