From Stdlib Require Import Arith.PeanoNat.
Theorem proof_bridge_target : forall n m : nat , S n + m = S ( n + m ).
Proof.
intros pb_local_0 pb_local_1.
induction pb_local_0 as [| pb_local_2 pb_local_3].
-
simpl.
reflexivity.
-
rewrite pb_local_3.
reflexivity.
Qed.
Check (proof_bridge_target : forall n m : nat , S n + m = S ( n + m )).
Print Assumptions proof_bridge_target.
