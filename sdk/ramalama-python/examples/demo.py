from ramalama_sdk import RamalamaModel

with RamalamaModel(model="gemma3:1b") as model:
    message = model.chat("hello")
