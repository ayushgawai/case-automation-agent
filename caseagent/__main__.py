import argparse
import json

from .evaluate import evaluate


def main():
    p = argparse.ArgumentParser(prog="caseagent")
    p.add_argument("--train", type=int, default=300)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--kb", default="kb")
    a = p.parse_args()
    print(json.dumps(evaluate(a.train, a.test, a.seed, a.kb), indent=2))


if __name__ == "__main__":
    main()
