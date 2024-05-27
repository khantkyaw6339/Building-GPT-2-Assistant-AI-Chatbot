# Building-GPT-2-Assistant-AI-Chatbot
This project involves developing a chatbot using a pretrained GPT-2 model. The model has been trained on the OASST1 dataset and is designed to to assist users with various tasks and queries.. The fine-tuned model is uploaded to Hugging Face under the name [KhantKyaw/Chat_GPT-2](https://huggingface.co/KhantKyaw/Chat_GPT-2). 

## Table of Contents

- Introduction
- Model 
- Dataset
- Installation
- Usage
- Fine-Tuning Process
- Contributing
- License

## Introduction
Welcome to the GPT-2 Assistant AI Chatbot project! This project showcases how to leverage the capabilities of GPT-2, a state-of-the-art language model developed by OpenAI, to build an intelligent chatbot. By fine-tuning GPT-2 on the Open Assistant Conversations (OASST1) dataset, we've created a model capable of understanding and responding to a wide array of user inputs, making it a versatile assistant.

## Model 
The GPT-2 model used in this project is a highly advanced text generation model known for its ability to produce coherent and contextually relevant sentences. The model has been fine-tuned specifically for conversational tasks, enabling it to provide meaningful and helpful responses. You can find the original pretrained model on Hugging Face under the name [openai-community/gpt2](https://huggingface.co/openai-community/gpt2).

## Dataset
The [OASST1 dataset](https://www.kaggle.com/datasets/snehilsanyal/oasst1?select=oasst1-val.csv) used for fine-tuning contains conversations that simulate interactions between a user and an assistant. The dataset is split into two parts:
- df_train.csv: Training dataset
- df_val.csv: Validation dataset
    
## Installation
To use the assistant chatbot, you'll need to install the required packages. You can do this using pip:

``` python
pip install transformers
pip install torch
```

## Usage
To use the fine-tuned chatbot model for generating responses:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_response(input_text):

    inputs = tokenizer(input_text, return_tensors="pt")
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=100,  # Adjusted max_length
        temperature=0.3,
        top_k=40,
        top_p=0.85,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        do_sample=True,
        use_cache=True,
    )

    full_generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    bot_response_start = full_generated_text.find('[Bot]') + len('[Bot]')
    bot_response = full_generated_text[bot_response_start:]

    last_period_index = bot_response.rfind('.')
    if last_period_index != -1:
        bot_response = bot_response[:last_period_index + 1]

    return bot_response.strip()


model_name = 'KhantKyaw/Chat_GPT-2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
response = generate_response(user_input)
print("Chatbot:", response)

```
## Fine-Tuning Process
The repository includes the script for fine-tuning the GPT-2 model. You can use Fine_Tuning_GPT2_Chatbot.ipynb to start the fine-tuning process.

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or find a bug, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

