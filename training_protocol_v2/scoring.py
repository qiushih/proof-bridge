"""Eight-row v2 selection. Historical six-row v1 selection stays untouched."""
from collections import Counter

DEV_IDS = [f'pd0{n}_{a}' for n in (1, 2, 3, 4) for a in ('A', 'B')]
STEPS = list(range(0, 61, 6))


def checked(rows):
    if Counter(r.get('dev_example_id') for r in rows) != Counter(DEV_IDS):
        raise ValueError('Exactly the eight frozen dev rows are required')
    for r in rows:
        if r.get('condition') != 'argument_' + r['dev_example_id'][-1]:
            raise ValueError('Only the matching argument condition is allowed')
    return rows


def metrics(rows):
    checked(rows)
    passed = lambda r: r['verification']['status'] == 'PASS'
    faithful = lambda r: passed(r) and r['mathematical_argument_fidelity']['status'] == 'FAITHFUL'
    triple = lambda r: faithful(r) and r['requested_proof_steps']['matched'] is True
    result = {}
    for subset, ids in [('original_six', DEV_IDS[:6]), ('new_two', DEV_IDS[6:]), ('all_eight', DEV_IDS)]:
        group = [r for r in rows if r['dev_example_id'] in ids]
        pairs = [[r for r in group if r['dev_example_id'].rsplit('_', 1)[0] == t]
                 for t in sorted({i.rsplit('_', 1)[0] for i in ids})]
        counts = {'verified': sum(passed(r) for r in group),
                  'verified_faithful': sum(faithful(r) for r in group),
                  'requested_step_matches': sum(r['requested_proof_steps']['matched'] is True for r in group),
                  'verified_requested_steps': sum(passed(r) and r['requested_proof_steps']['matched'] is True for r in group),
                  'verified_faithful_requested_steps': sum(triple(r) for r in group)}
        result[subset] = {'rows': len(group), **counts,
                         'percentages': {k: 100 * v / len(group) for k, v in counts.items()},
                         'both_argument_successes': sum(all(triple(r) for r in g) for g in pairs),
                         'theorem_pairs': len(pairs)}
    return result


def selection_score(rows, optimizer_step):
    if type(optimizer_step) is not int or optimizer_step not in STEPS:
        raise ValueError('Checkpoint must be one of the predeclared steps')
    m = metrics(rows)['all_eight']
    return [m['verified_faithful'], m['verified_requested_steps'], m['verified'], -optimizer_step]


def success_criteria(control, intervention):
    c, t = metrics(control), metrics(intervention)
    a, b = c['all_eight'], t['all_eight']
    gates = {
        'strictly_more_verified_faithful_requested_steps': b['verified_faithful_requested_steps'] > a['verified_faithful_requested_steps'],
        'verification_not_lower_than_control': b['verified'] >= a['verified'],
        'mathematical_fidelity_not_lower_than_control': b['verified_faithful'] >= a['verified_faithful'],
        'original_six_preserve_verification_and_fidelity': t['original_six']['verified_faithful'] == 6,
        'both_new_arguments_verified_faithful_and_step_adherent': t['new_two']['both_argument_successes'] == 1,
    }
    return {'success': all(gates.values()), 'gates': gates,
            'interpretation': 'Predeclared development decision only; one seed and correlated theorem pairs.'}
