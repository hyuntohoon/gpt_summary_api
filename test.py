import openai

openai.api_key = "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"

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])
print(completion.choices[0].message)
