import sys
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

checkpoint_dir = sys.argv[1]
steps = sys.argv[2]

print('='*80)
print('Waking up the model...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained(checkpoint_dir)

print('='*80)
print("Why don't you say hello?")
chat_history_ids = torch.tensor([], dtype=torch.long)
for step in range(int(steps)):
    user_input = input('>>> User: ')
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    # truncate the input ids to last 250
    bot_input_ids = bot_input_ids[:, -249:]
    chat_history_ids = model.generate(
        bot_input_ids, max_length=250,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature = 0.8
    )
    print(">>> Yoni: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
