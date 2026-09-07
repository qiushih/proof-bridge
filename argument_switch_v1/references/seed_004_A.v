From Stdlib Require Import Arith.PeanoNat.
Theorem proof_bridge_target : forall n : nat , 1 + n = S n.
Proof.
intros pb_local_0.
simpl.
reflexivity.
Qed.
Check (proof_bridge_target : forall n : nat , 1 + n = S n).
Print Assumptions proof_bridge_target.
