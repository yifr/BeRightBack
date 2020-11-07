import io
import os 
import json
from tqdm import tqdm
import pandas as pd

def get_conversations(data_dir):
    if not os.path.exists(data_dir):
        raise NotADirectoryError
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                fpath = os.path.join(dirpath, filename)
                with open(fpath, encoding='utf-16') as f:
                    conversation = json.load(f)
                    yield(conversation)


def json_to_df(conversation, target_name, df):
    """
    Output messages from conversation json to dataframe
    Params:
        conversation: json dictionary of converation
        target_name: name of person to model
        df: dataframe to store messages in
    """
    # Ignore groupchats
    if len(conversation.get('participants', [])) > 2:
        return df
    
    talking_to = conversation['participants'][0] 
    message = ''
    response = ''
    for text in reversed(conversation.get('messages')):
        # Ignore non-text messages
        if text.get('type') != 'Generic':
            continue

        if text.get('sender_name') == target_name:
            if message and text.get('content'):
                if len(response) > 0:
                    text['content'] = '\n' + text['content']
                response += text['content']
        else:
            if len(response) > 0:
                # If we've already responded, log new interaction and reset
                interaction = {'message': message, 'response': response, 'talking_to': talking_to}
                df = df.append(interaction, ignore_index=True)
                response = ''
                message = ''

            if text.get('content'):
                if len(message) > 0:
                    text['content'] = ' ' + text['content']
                message += text['content']
        
    return df

def process_data(data_dir='data/inbox', target='Yoni Friedman', outfile='data/data_clean.csv'):
    df = pd.DataFrame(columns=['message', 'response', 'talking_to'])

    for conversation in tqdm(get_conversations(data_dir)):
        df = json_to_df(conversation, target, df)

    print(df.info())
    df.to_csv(outfile)
    
    return df