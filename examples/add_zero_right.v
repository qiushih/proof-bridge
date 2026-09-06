(* Proof Bridge milestone 1: compiled with Rocq 9.2.0 / Stdlib 9.1.0.
   Re-run scripts/verify_example.py to refresh the JSON verification record. *)

From Stdlib Require Import Arith.PeanoNat.

Theorem add_zero_right : forall n : nat, n + 0 = n.
Proof.
  intros n.
  induction n as [| k IH].
  - simpl.
    reflexivity.
  - simpl.
    rewrite IH.
    reflexivity.
Qed.

Print Assumptions add_zero_right.
