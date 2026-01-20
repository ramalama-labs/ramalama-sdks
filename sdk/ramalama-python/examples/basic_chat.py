from ramalama_sdk import RamalamaModel

with RamalamaModel(model="tinyllama") as model:
    response = model.chat("How tall is Michael Jordan?")
    print(response["content"])
