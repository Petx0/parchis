import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAMPION = "runs/selfplay_2p_v1_champion/model.pt"


def setup():
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def load_champion():
    setup()
    from parchis.az import champion_pool
    return champion_pool.load_numpy_net(CHAMPION, 216, 2, (256, 256))
