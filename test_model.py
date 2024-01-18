
from transformers import AutoModel, AutoTokenizer

text = "UDP-alpha-D-galactofuranose(2-) is a UDP-D-galactofuranose(2-) in which the anomeric centre of the galactofuranose moiety has alpha-configuration. It is a conjugate base of an UDP-alpha-D-galactofuranose."

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp = AutoModel.from_pretrained(model_name)

text_input = tokenizer([text], return_tensors="pt", max_length=128, truncation=True, add_special_tokens=True)
output = nlp(**text_input)
print(output.last_hidden_state.shape)
