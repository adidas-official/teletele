import ollama

response = ollama.chat(
    model='llama3',
    messages=[
        {
            'role': 'user',
            'content': 'Ahoj! Stručně vysvětli, co je to Docker.',
        },
    ]
)

print(response['message']['content'])