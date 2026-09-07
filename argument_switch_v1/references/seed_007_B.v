From Stdlib Require Import Arith.PeanoNat.
Theorem proof_bridge_target : forall n m : nat , n = m -> n = m.
Proof.
intros pb_local_0 pb_local_1 pb_local_2.
induction pb_local_0 as [| pb_local_3 pb_local_4].
-
simpl.
exact pb_local_2.
-
simpl.
exact pb_local_2.
Qed.
Check (proof_bridge_target : forall n m : nat , n = m -> n = m).
Print Assumptions proof_bridge_target.
