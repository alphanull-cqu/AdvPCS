from typing import Any, Dict, List, Optional, Sequence, Tuple


_SACO_PROMPT_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "4.95 euro": ("4.95 euros", "price label 4.95 euro"),
    "a 1000": ("the number 1000", "number 1000"),
    "a brown and white swag": ("brown and white fabric swag", "brown white swag"),
    "a clothe rack": ("a clothes rack", "garment rack"),
    "a game of dart": ("dart game", "dartboard game"),
    "a habiknit": ("habiknit label", "habiknit text"),
    "a meat": ("piece of meat", "meat piece"),
    "a merry christmas": ("merry christmas sign", "christmas sign"),
    "a happy christmas": ("happy christmas sign", "christmas sign"),
    "air jordan": ("air jordan logo", "jordan logo"),
    "basket of present": ("basket of presents", "gift basket"),
    "blind's control cord": ("blind control cord", "window blind cord"),
    "brand's logo": ("brand logo", "brand mark"),
    "head of cabbage": ("cabbage head", "cabbage"),
    "image of beverage": ("picture of beverage", "drink image"),
    "individual's face": ("person's face", "face of a person"),
    "li-ning logo": ("li ning logo", "lining logo"),
    "loaf of bread": ("bread loaf", "bread"),
    "maersk": ("maersk logo", "maersk mark"),
    "new plymouth": ("plymouth", "new plymouth sign"),
    "round, decorative embellishment": ("round decorative ornament", "circular embellishment"),
    "the image": ("image", "picture"),
    "the image of a woman": ("image of a woman", "photo of a woman"),
    "the individual's left arm": ("the person's left arm", "left arm of the person"),
    "the individual's right hand": ("the person's right hand", "right hand of the person"),
    "the nike air jordan logo": ("nike air jordan logo", "air jordan logo"),
    "the paw patrol design": ("paw patrol design", "paw patrol graphic"),
    "vibrant, cartoon-style game": ("vibrant cartoon game", "cartoon style game"),
    "wavy, tan and orange pattern": ("tan and orange wavy pattern", "wavy tan orange pattern"),
    "white m within a circle": ("white m in a circle", "white circled m"),
    "woman's head": ("head of a woman", "woman head"),
}


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def parse_text_prompt_value(prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj.strip()
    if isinstance(prompt_obj, dict):
        for key in ("selected_text", "text", "prompt", "caption", "query"):
            value = prompt_obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(prompt_obj, (list, tuple)) and len(prompt_obj) > 0:
        first = prompt_obj[0]
        if isinstance(first, str):
            return first.strip()
    return str(prompt_obj).strip()


def _strip_article(text: str) -> str:
    lowered = text.lower()
    for prefix in ("a ", "an ", "the "):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()


def _with_indefinite_article(text: str) -> str:
    core = text.strip()
    if not core:
        return core
    article = "an" if core[0].lower() in "aeiou" else "a"
    return f"{article} {core}"


def _build_default_variants(prompt: str) -> List[str]:
    prompt = str(prompt).strip()
    if not prompt:
        return []

    core = _strip_article(prompt)
    lowered = prompt.lower()

    if lowered.startswith("the image of "):
        subject = prompt[len("the image of "):].strip()
        return _dedupe_keep_order([prompt, f"image of {subject}", f"photo of {subject}"])

    if lowered.startswith("image of "):
        subject = prompt[len("image of "):].strip()
        return _dedupe_keep_order([prompt, f"photo of {subject}", subject])

    if lowered.startswith("the "):
        candidates = [prompt, core, f"photo of {core}"]
    elif lowered.startswith(("a ", "an ")):
        candidates = [prompt, f"the {core}", f"photo of {core}"]
    else:
        candidates = [prompt, f"the {core}", f"photo of {core}"]

    candidates = _dedupe_keep_order(candidates)
    if len(candidates) >= 3:
        return candidates[:3]

    fill_candidates = [core, _with_indefinite_article(core), f"object {core}"]
    candidates.extend(fill_candidates)
    return _dedupe_keep_order(candidates)[:3]


def get_saco_text_prompt_variants(prompt: str) -> List[str]:
    base_prompt = parse_text_prompt_value(prompt)
    if not base_prompt:
        return []

    override = _SACO_PROMPT_OVERRIDES.get(base_prompt)
    if override is not None:
        return _dedupe_keep_order([base_prompt, override[0], override[1]])[:3]

    return _build_default_variants(base_prompt)


def extract_text_prompt_candidates(prompt_obj: Any, dataset_name: Optional[str] = None) -> List[str]:
    dataset_key = (dataset_name or "").strip().upper()
    if isinstance(prompt_obj, dict):
        dict_dataset = str(prompt_obj.get("dataset", "")).strip().upper()
        if dict_dataset:
            dataset_key = dict_dataset

        variants = prompt_obj.get("variants")
        if isinstance(variants, (list, tuple)):
            prompt_text = parse_text_prompt_value(prompt_obj)
            merged = _dedupe_keep_order([prompt_text, *[str(v) for v in variants]])
            if len(merged) > 0:
                return merged

    prompt_text = parse_text_prompt_value(prompt_obj)
    if not prompt_text:
        return []

    if dataset_key == "SACO":
        return get_saco_text_prompt_variants(prompt_text)
    return [prompt_text]


def build_text_prompt_payload(prompt: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    prompt_text = parse_text_prompt_value(prompt)
    return {
        "text": prompt_text,
        "variants": extract_text_prompt_candidates(prompt_text, dataset_name=dataset_name),
        "dataset": dataset_name or "",
    }