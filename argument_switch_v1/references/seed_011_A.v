From Stdlib Require Import Arith.PeanoNat.
Theorem proof_bridge_target : forall n m p : nat , n = m -> p + n = p + m.
Proof.
intros pb_local_0 pb_local_1 pb_local_2 pb_local_3.
rewrite pb_local_3.
reflexivity.
Qed.
Check (proof_bridge_target : forall n m p : nat , n = m -> p + n = p + m).
Print Assumptions proof_bridge_target.
