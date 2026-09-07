"""Token-level masks from the character machine and the pinned tokenizer."""

from collections import OrderedDict
import time

from constrained_v1.grammar import ALPHABET, advance, can_end, feed


class Vocabulary:
    def __init__(self, tokenizer):
        self.eos_id = tokenizer.eos_token_id
        self.size = len(tokenizer)
        self.root = {}
        self.pieces = {}
        specials = set(tokenizer.all_special_ids)
        for token_id in range(self.size):
            if token_id in specials:
                continue
            piece = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not piece or any(c not in ALPHABET for c in piece):
                continue
            self.pieces[token_id] = piece
            node = self.root
            for char in piece:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(token_id)
        self.cache = OrderedDict()

    def allowed(self, state):
        cached = self.cache.get(state)
        if cached is not None:
            self.cache.move_to_end(state)
            return cached
        ids = []
        stack = [(self.root, state)]
        while stack:
            node, current = stack.pop()
            ids.extend(node.get(None, ()))
            for char, child in node.items():
                if char is None:
                    continue
                following = advance(current, char)
                if following is not None:
                    stack.append((child, following))
        if can_end(state):
            ids.append(self.eos_id)
        result = tuple(ids)
        self.cache[state] = result
        if len(self.cache) > 1024:
            self.cache.popitem(last=False)
        return result


class Mask:
    """Hugging Face logits processor; one sequence, greedy, no beams or repair."""
    def __init__(self, vocabulary, state, prompt_length):
        self.vocabulary = vocabulary
        self.state = state
        self.prompt_length = prompt_length
        self.consumed = 0
        self.trace = []
        self.mask_seconds = 0.0
        self.seen_ids = []

    def __call__(self, input_ids, scores):
        started = time.perf_counter()
        if input_ids.shape[0] != 1:
            raise ValueError("Only one greedy generation is supported")
        produced = input_ids[0, self.prompt_length:].tolist()
        self.seen_ids = produced
        for token_id in produced[self.consumed:]:
            piece = self.vocabulary.pieces.get(token_id)
            if piece is None:
                raise ValueError("Unexpected special/non-ASCII generated token")
            self.state = feed(self.state, piece)
            if self.state is None:
                raise ValueError("Generated prefix escaped the grammar")
        self.consumed = len(produced)
        allowed = self.vocabulary.allowed(self.state)
        if not allowed:
            raise ValueError("No valid continuation; refusing an unconstrained fallback")
        raw_top = int(scores[0].argmax())
        masked = scores.new_full(scores.shape, float("-inf"))
        masked[0, list(allowed)] = scores[0, list(allowed)]
        self.trace.append({"position": len(produced), "allowed_tokens": len(allowed),
                           "unconstrained_top_token_id": raw_top, "top_token_blocked": raw_top not in allowed})
        self.mask_seconds += time.perf_counter() - started
        return masked
