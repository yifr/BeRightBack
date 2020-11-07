# BeRightBack

We'll be using Hugging Face's DialoGPT model to train a neural chat bot on data extracted from my facebook messenger data.
If you have access to your messenger data (in json format) you can run this code by simply running 

`python fit.py`

Once your model is trained, you can use the `run_conversation.py <checkpoint_directory> <n_conversation_lines>` function to interact with your bot
