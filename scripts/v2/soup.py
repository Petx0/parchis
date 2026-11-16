"""Weight soup of AZNet checkpoints (PILA-8, parchis/az/soup.py).

Usage: soup.py <out.pt> <a.pt> <b.pt> [<c.pt> ...] [weights=w1,w2,...]

Averages the state dicts (uniformly unless weights= is given; weights are normalised to sum to 1),
asserts that keys, shapes and dtypes match, and writes <out.pt> plus config.json in the same
directory listing the ingredients with their sha256 and weights. Light: no games, no lock.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main(argv):
    positional = [a for a in argv if "=" not in a]
    extra = dict(a.split("=", 1) for a in argv if "=" in a)
    unknown = set(extra) - {"weights"}
    if len(positional) < 3 or unknown:
        print(__doc__, file=sys.stderr)
        return 4
    out, ingredients = positional[0], positional[1:]
    weights = [float(w) for w in extra["weights"].split(",")] if "weights" in extra else None
    from parchis.az import soup
    config = soup.make_soup(out, ingredients, weights)
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
