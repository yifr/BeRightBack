import os 
import json
import pandas as pd

def get_messages(data_dir):
    if not os.path.exists(data_dir):
        raise NotADirectoryError
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                path = os.path.join(dirpath, filename)
                with open(path) as p:
                    message = json.load(p)
                    yield(message)

    

if __name__=='__main__':
    i = 0
    for p in get_messages('data/inbox'):
        print(p)
        i += 1
        if i == 100:
            break