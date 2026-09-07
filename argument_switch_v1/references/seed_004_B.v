From Stdlib Require Import Arith.PeanoNat.
Theorem proof_bridge_target : forall n : nat , 1 + n = S n.
Proof.
intros pb_local_0.
induction pb_local_0 as [| pb_local_1 pb_local_2].
-
simpl.
reflexivity.
-
rewrite <- pb_local_2.
reflexivity.
Qed.
Check (proof_bridge_target : forall n : nat , 1 + n = S n).
Print Assumptions proof_bridge_target.
