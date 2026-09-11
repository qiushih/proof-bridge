"""Controlled reference authoring before sealing; never model-generated targets."""
from pilot_v1.protocol import features

TRAIN_SPEC = {
    'theorem_id': 'ptv201', 'split': 'train', 'generation_family': 'right_zero_variants',
    'formal_statement': 'forall n m p : nat, m = p -> (n + 0) + m = n + p',
    'informal_statement': 'If m equals p, then adding m to n + 0 gives n + p, for every natural number n.',
    'premise': 'm = p', 'base': 'm = p',
    'ih': '(k + 0) + m = k + p',
    'successor_goal': 'S ((k + 0) + m) = S (k + p)',
    'review_notes': 'The right-zero residual is inside a larger sum with a variable suffix, unlike the original premise-base rows ending in + 0. Induction follows its recursive left operand; the base premise and IH are useful in the script. Substituting the premise alone leaves a right-zero obligation. This remains one existing training family, related to historical seed_021, not an independent new family.',
}


def make_pair(spec):
    rows = []
    for argument in ('A', 'B'):
        steps = [
            {'text': f"Fix natural numbers n, m and p, and assume {spec['premise']}.", 'code': 'intros n m p H.'},
            {'text': 'Proceed by induction on n, keeping m, p and the premise fixed.', 'code': 'induction n as [| k IH].'},
            {'text': f"At n = 0, simplification gives {spec['base']}. Use the premise to close this base case.", 'code': '- simpl.\n  exact H.'},
            {'text': f"For the successor case, assume {spec['ih']}. At n = S k, simplification gives {spec['successor_goal']}.", 'code': '- simpl.'},
            ({'text': 'Rewrite using the induction hypothesis from left to right inside the successor. The resulting sides are identical.', 'code': '  rewrite IH.\n  reflexivity.'}
             if argument == 'A' else
             {'text': 'Apply successor congruence to reduce the goal to equality of the inner terms. Use the induction hypothesis to finish.', 'code': '  f_equal.\n  exact IH.'}),
        ]
        body = '\n'.join(s['code'] for s in steps)
        rows.append({
            'schema_version': 'pilot-2.0', 'id': spec['theorem_id'] + '_' + argument,
            'theorem_id': spec['theorem_id'], 'argument_id': argument,
            'split': spec['split'], 'generation_family': spec['generation_family'],
            'formal_statement': spec['formal_statement'], 'informal_statement': spec['informal_statement'],
            'informal_proof': ' '.join(s['text'] for s in steps), 'proof_body': body, 'steps': steps,
            'argument_features': features(spec['formal_statement'], body),
            'contrast_type': 'equivalent_ih_realization',
            'argument_contract': {'induction_variable': 'n', 'base': 'premise',
                                  'successor_ih_use': 'rewrite' if argument == 'A' else 'congruence',
                                  'ih_direction': 'forward' if argument == 'A' else None},
            'review': {'reviewer': 'assistant', 'human_reviewed': False, 'primary_target_eligible': True,
                      'alignment': 'assistant-reviewed', 'unnecessary_induction': False,
                      'redundant_ih_use': False, 'notes': spec['review_notes']},
            'provenance': {'informal_statement_source': 'assistant-curated', 'informal_proof_source': 'assistant-curated',
                           'formal_proof_source': 'assistant-curated', 'human_reviewed': False,
                           'rocq_verified': False, 'model_generated': False, 'historical_reference_copied': False,
                           'curation': 'Explicitly authored statement and aligned A/B steps; no paraphrase enumeration.',
                           'source_experiments': ['pilot-v2-coverage-audit', 'v2-dev-baseline']},
        })
    return rows
