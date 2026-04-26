def select_qualitative_rows(df, n_correct=4, n_wrong=2):
    correct = df[df.correct].head(n_correct)
    wrong = df[~df.correct].head(n_wrong)
    return list(correct.to_dict("records")) + list(wrong.to_dict("records"))

