from config import TRAIN_SIZE, TEST_SIZE, STEP


def rolling_split(df, train_size=TRAIN_SIZE, test_size=TEST_SIZE, step=STEP):
    splits = []
    n = len(df)

    for start in range(0, n - train_size - test_size + 1, step):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        splits.append((train, test))

    return splits