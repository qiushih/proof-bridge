"""Incremental ASCII grammar with theorem-conditioned local scope.

Only the theorem's binder counts enter the machine. No reference argument,
reference proof, theorem family, or compiler feedback enters token selection.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
import string

from verifier import MAX_COMMANDS, RESERVED, StatementParser, tokenize

WORD_START = frozenset(string.ascii_letters)
WORD = WORD_START | frozenset(string.digits + "_'")
SPACE = frozenset(" \n\t\r")
ALPHABET = WORD | SPACE | frozenset(".[]|<->")
NO_ARGS = ("simpl", "reflexivity", "symmetry", "f_equal")


@dataclass(frozen=True, slots=True)
class State:
    nat_count: int
    binder_count: int
    phase: str = "command"
    scope: tuple = ()  # (name, nat|eq|ih)
    original_scope: tuple = ()
    introduced: int = 0
    command_count: int = 0
    intro_args: int = 0
    induction_target: str = ""
    successor_name: str = ""
    ih_name: str = ""
    branch: int = 0
    branch_commands: int = 0
    proof_commands: int = 0
    buffer: str = ""
    separator_required: bool = False


def initial(statement):
    parser = StatementParser(tokenize(statement))
    _, count = parser.parse()
    return State(nat_count=len(parser.variables), binder_count=count)


def free_name(state):
    return state.phase in ("intro_arg", "successor_name", "ih_name")


def valid_name(state, name):
    occupied = {n for n, _ in state.scope} | {n for n, _ in state.original_scope}
    if state.successor_name:
        occupied.add(state.successor_name)
    return (1 <= len(name) <= 32 and name[0] in WORD_START and all(c in WORD for c in name)
            and name not in RESERVED and not name.startswith("pb_local_") and name not in occupied)


def literals(state):
    p = state.phase
    if p == "command":
        if state.command_count >= MAX_COMMANDS:
            return ()
        if state.branch == 1 and state.command_count == MAX_COMMANDS - 1:
            return ("-",) if state.branch_commands else ()
        if state.introduced < state.binder_count:
            return ("intros",)
        result = list(NO_ARGS)
        if any(kind in ("eq", "ih") for _, kind in state.scope):
            result += ["rewrite", "exact", "apply"]
        # Reserve enough command slots for one tactic in each induction branch.
        if not state.induction_target and state.command_count <= MAX_COMMANDS - 3:
            result.append("induction")
        if state.branch == 1 and state.branch_commands:
            result.append("-")
        return tuple(result)
    if p == "intro_arg":
        return (".",) if state.intro_args else ()
    if p == "intro_end":
        return (".",)
    if p == "induction_target":
        return tuple(n for n, kind in state.scope if kind == "nat")
    if p == "as":
        return ("as",)
    if p == "left_bracket":
        return ("[",)
    if p == "bar":
        return ("|",)
    if p == "right_bracket":
        return ("]",)
    if p in ("period", "induction_period"):
        return (".",)
    if p == "first_branch":
        return ("-",)
    if p in ("reference", "rewrite_reference"):
        refs = tuple(n for n, kind in state.scope if kind in ("eq", "ih"))
        return (("<-", "->") + refs) if p == "rewrite_reference" else refs
    return ()


def viable_word(state, word):
    if any(value.startswith(word) for value in literals(state)):
        return True
    if free_name(state) and word[0] in WORD_START and all(c in WORD for c in word):
        if word.startswith("pb_local_") or len(word) > 32:
            return False
        return len(word) < 32 or valid_name(state, word)
    return False


def commit(state, token):
    p = state.phase
    if not (token in literals(state) or (free_name(state) and valid_name(state, token))):
        return None
    if p == "command":
        if token == "-":
            scope = tuple(x for x in state.original_scope if x[0] != state.induction_target)
            scope += ((state.successor_name, "nat"), (state.ih_name, "ih"))
            return replace(state, scope=scope, branch=2, branch_commands=0)
        if token == "intros":
            return replace(state, phase="intro_arg", intro_args=0)
        if token == "induction":
            return replace(state, phase="induction_target")
        if token in NO_ARGS:
            return replace(state, phase="period")
        return replace(state, phase="rewrite_reference" if token == "rewrite" else "reference")
    if p == "intro_arg" and token != ".":
        kind = "nat" if state.introduced < state.nat_count else "eq"
        count = state.introduced + 1
        return replace(state, scope=state.scope + ((token, kind),), introduced=count,
                       intro_args=state.intro_args + 1,
                       phase="intro_end" if count == state.binder_count else "intro_arg")
    if p in ("intro_arg", "intro_end") and token == ".":
        return replace(state, phase="command", command_count=state.command_count + 1,
                       separator_required=True)
    if p == "induction_target":
        return replace(state, phase="as", induction_target=token, original_scope=state.scope)
    next_phase = {"as": "left_bracket", "left_bracket": "bar", "bar": "successor_name",
                  "right_bracket": "induction_period"}
    if p in next_phase:
        return replace(state, phase=next_phase[p])
    if p == "successor_name":
        return replace(state, phase="ih_name", successor_name=token)
    if p == "ih_name":
        return replace(state, phase="right_bracket", ih_name=token)
    if p == "induction_period":
        return replace(state, phase="first_branch", command_count=state.command_count + 1,
                       separator_required=True)
    if p == "first_branch":
        scope = tuple(x for x in state.original_scope if x[0] != state.induction_target)
        return replace(state, phase="command", branch=1, scope=scope, branch_commands=0)
    if p == "rewrite_reference" and token in ("<-", "->"):
        return replace(state, phase="reference")
    if p in ("reference", "rewrite_reference"):
        return replace(state, phase="period")
    if p == "period":
        # A base branch must leave a command slot for the successor branch.
        count = state.command_count + 1
        if count >= MAX_COMMANDS and state.branch == 1:
            return None
        return replace(state, phase="command", command_count=count,
                       proof_commands=state.proof_commands + 1,
                       branch_commands=state.branch_commands + (state.branch > 0),
                       separator_required=True)
    raise AssertionError((p, token))


@lru_cache(maxsize=250_000)
def advance(state, char):
    if char not in ALPHABET:
        return None
    if char in SPACE:
        if state.buffer:
            state = commit(replace(state, buffer=""), state.buffer)
            if state is None:
                return None
        return replace(state, separator_required=False)
    if state.separator_required:
        return None
    if state.buffer:
        word_buffer = state.buffer[0] in WORD_START
        if word_buffer and char in WORD:
            word = state.buffer + char
            return replace(state, buffer=word) if viable_word(state, word) else None
        if not word_buffer:
            candidate = state.buffer + char
            options = literals(state)
            if candidate in options:
                return commit(replace(state, buffer=""), candidate)
            if any(value.startswith(candidate) for value in options):
                return replace(state, buffer=candidate)
        updated = commit(replace(state, buffer=""), state.buffer)
        return advance(updated, char) if updated is not None else None
    if char in WORD_START:
        return replace(state, buffer=char) if viable_word(state, char) else None
    options = literals(state)
    if char in options:
        return commit(state, char)
    if any(value.startswith(char) for value in options):
        return replace(state, buffer=char)
    return None


def feed(state, text):
    for char in text:
        state = advance(state, char)
        if state is None:
            return None
    return state


def can_end(state):
    return bool(state is not None and not state.buffer and state.phase == "command"
                and state.introduced == state.binder_count and state.proof_commands
                and (not state.induction_target or (state.branch == 2 and state.branch_commands)))


def accepts(statement, body):
    return can_end(feed(initial(statement), body))
