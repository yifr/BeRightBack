import os
import numpy as np
import logging
import pickle
import torch
from data import process_data
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
 
logger = logging.getLogger(__name__)


class Conversations(Dataset):
    def __init__(self, conversation_df, tokenizer, args, block_size=512):
        self.tokenizer = tokenizer
        self.records = []

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + '_cached_lm_' + str(block_size)
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info('Loading features from cached file %s' % cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.records = pickle.load(handle)
        else:
            logger.info('Creating features from dataset file at %s' % directory)
            self.records = []
            n_context = 0
            for i in range(n_context, len(conversation_df.index)):
                record = self.create_record(i, conversation_df, n_context=n_context)
                self.records.append(record)
            
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.records, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def create_record(self, idx, df, n_context=6):
        """
        Format record as response + n_context - 1 messages of previous context.
        Between each message we'll append an "end of sentence" token
        return tokenized record
        """
        tokenizer = self.tokenizer        
        resp = df['response'].iloc[idx]
        msg = df['message'].iloc[idx]
        record = tokenizer.encode(msg) + [tokenizer.eos_token_id] + tokenizer.encode(resp) + [tokenizer.eos_token_id]
        return record

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return torch.tensor(self.records[index], dtype=torch.long)

def get_train_test(tokenizer, args, data_dir='data/inbox', target='Yoni Friedman', outfile='data/data_clean.csv'):
    all_data = process_data.process_data(data_dir=data_dir, target=target, outfile=outfile)
    
    train_df, test_df = train_test_split(all_data, test_size=0.2)
    train_features = Conversations(train_df, tokenizer, args)
    test_features = Conversations(test_df, tokenizer, args)

    return train_features, test_features