import subprocess
import re
from typing import Dict, List, Tuple


# ==========================================================
# Command dictionary  (4-bit opcode -> ABC macro)
# ==========================================================
COMMAND_DICT: Dict[int, str] = {
    0b0000: "rewrite",
    0b0001: "rewrite -z",
    0b0010: "refactor",
    0b0011: "refactor -z",
    0b0100: "resub",
    0b0101: "resub -z",
    0b0110: "balance",
    0b0111: "ifraig",
    0b1000: "dfraig",
    0b1001: "rewrite",
    0b1010: "refactor",
    0b1011: "balance",
    0b1100: "&get -n; &sopb; &put",
    0b1101: "&get -n; &blut; &put",
    0b1110: "&get -n; &dsdb; &put",
    0b1111: "&get -n; &mfs; &put",
}


# ==========================================================
# 1) Genome bits -> ABC optimisation script
# ==========================================================
def bits_to_script(bits: List[int]) -> str:
    """
    Convert a flat list of bits (length must be a multiple of 4)
    into a semicolon-separated ABC command string.
    """
    if len(bits) % 4 != 0:
        raise ValueError(f"bits length must be a multiple of 4, got {len(bits)}")

    commands: List[str] = []
    for i in range(0, len(bits), 4):
        chunk = bits[i : i + 4]
        opcode = (chunk[0] << 3) | (chunk[1] << 2) | (chunk[2] << 1) | chunk[3]
        commands.append(COMMAND_DICT.get(opcode, f"#ERR:{opcode}"))

    return "; ".join(commands) + ";"


# ==========================================================
# 2) Parse ABC print_stats output
# ==========================================================
def parse_stats(stdout: str) -> Tuple[float, float]:
    """
    Extract LUT count (nd) and logic depth (lev) from ABC stdout.
    Returns (inf, inf) when either metric is missing.
    """
    lut = float("inf")
    level = float("inf")

    m_nd = re.search(r"nd\s*=\s*([0-9]+)", stdout)
    if m_nd:
        lut = float(m_nd.group(1))

    m_lev = re.search(r"lev\s*=\s*([0-9]+)", stdout)
    if m_lev:
        level = float(m_lev.group(1))

    return lut, level


# ==========================================================
# 3) Run a single ABC evaluation
# ==========================================================
def evaluate_design(
    design_file: str,
    bits: List[int],
    lut_k: int,
    abc_bin: str = "yosys-abc",
) -> Tuple[float, float]:
    """
    Run ABC on *design_file* using the optimisation script encoded
    in *bits*, then map to LUTs with K = *lut_k*.

    Returns:
        (lut_count, logic_depth)

    Raises:
        RuntimeError  – if the ABC binary is missing or exits non-zero.
    """
    script = (
        f"read {design_file}; "
        "strash; "
        f"{bits_to_script(bits)} "
        f"if -K {lut_k}; "
        "print_stats;"
    )

    try:
        proc = subprocess.run(
            [abc_bin, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ABC binary not found: {abc_bin}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"ABC exited with code {proc.returncode}\n"
            f"SCRIPT :\n{script}\n"
            f"STDERR :\n{proc.stderr}\n"
            f"STDOUT :\n{proc.stdout}"
        )

    return parse_stats(proc.stdout)


# ==========================================================
# 4) Fitness / QoR computation
# ==========================================================
def compute_fitness(
    ref_lut: float,
    ref_level: float,
    lut: float,
    level: float,
) -> Tuple[float, float]:
    """
    Normalise metrics against the reference and return:
        qor     : combined quality-of-result score (lower = better; baseline = 2.0)
        improve : improvement over baseline in percent
    """
    if ref_lut <= 0 or ref_level <= 0:
        raise ValueError("Reference metrics must be positive")

    qor = (lut / ref_lut) + (level / ref_level)
    improve = (2.0 - qor) / 2.0 * 100.0

    return qor, improve

