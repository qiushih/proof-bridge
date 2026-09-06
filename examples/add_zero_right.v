(* Illustrative target for the initial Proof Bridge plan.
   Reviewed against documented Rocq induction patterns, but not yet
   compiled in this workspace. See RESEARCH_PLAN.md for verification status. *)

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
