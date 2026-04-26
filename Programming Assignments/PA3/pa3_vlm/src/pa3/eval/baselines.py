from collections import Counter
from pa3.common.metrics import normalize_answer


def majority_accuracy(rows):
    answers = [normalize_answer(r["answer"]) for r in rows]
    if not answers:
        return {"answer": "", "acc": 0.0}
    answer, count = Counter(answers).most_common(1)[0]
    return {"answer": answer, "acc": count / len(answers)}

