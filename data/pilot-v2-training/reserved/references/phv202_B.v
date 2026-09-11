From Stdlib Require Import Arith.PeanoNat.
Theorem proof_bridge_target : forall n m p : nat , m + ( m + m ) = p -> ( n + m ) + ( m + m ) = n + p.
Proof.
intros pb_local_0 pb_local_1 pb_local_2 pb_local_3.
induction pb_local_0 as [| pb_local_4 pb_local_5].
-
simpl.
exact pb_local_3.
-
simpl.
f_equal.
exact pb_local_5.
Qed.
Check (proof_bridge_target : forall n m p : nat , m + ( m + m ) = p -> ( n + m ) + ( m + m ) = n + p).
Print Assumptions proof_bridge_target.
