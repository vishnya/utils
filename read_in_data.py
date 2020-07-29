def print_local_data(data_dir):
    import os
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def read_data_in_chunks_into_df(chunksize,
                                path_to_datafile,
                                windows=True,
                                sample_frac=1,
                                sample_random_state=123):
    import pandas as pd
    df = pd.DataFrame()
    num_of_chunks = 0
    if windows:
        path_to_datafile.replace('/', '//')
    for chunk in pd.read_csv(path_to_datafile,
                             chunksize=chunksize):
        num_of_chunks += 1
        df = pd.concat([df, chunk.sample(frac=sample_frac,
                                         replace=False,
                                         random_state=sample_random_state)],
                       axis=0)
        print('Processing Chunk No. ' + str(num_of_chunks))

    df.reset_index(inplace=True)
