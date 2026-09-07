From Stdlib Require Import Arith.PeanoNat.
Theorem proof_bridge_target : forall n m : nat , ( 0 + n ) + m = n + m.
Proof.
intros pb_local_0 pb_local_1.
simpl.
reflexivity.
Qed.
Check (proof_bridge_target : forall n m : nat , ( 0 + n ) + m = n + m).
Print Assumptions proof_bridge_target.
