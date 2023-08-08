from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

prompt = """
The following is a conversation between a customer and an AI product expert. The customer will provide a product description and the AI product expert analyses, comprehend and understands it. The customer will ask a question based on the given product description and the AI product always comes up with a detailed and helpful answer. The customer's question starts with <human>: and the AI assistant's answer starts with <assistant>:.
"BAND-AID Bandages Comfort-Flex Sheer Extra Large All One Size 10 Each (Pack of 6). Free ShippingPack of 6Brand: Band Aid. It is manufactured by Band Aid. Heals The Hurt Faster™.
Inside every sheer strips bandage you'll find these unique technologies:
Non-stick pad.
Hurt-Free™ pad that won't stick to the wound for gentle removal.
Enhanced coverage.
Extra-large pad designed to fit more wounds.
A covered wound heals faster than an uncovered one.
For medical emergencies seek professional help.
Sterile unless individual wrapper is opened or damaged.
FSC®.
www.fsc.org.
Mix.
Packaging from responsible sources.
Band-Aid® is a registered trademark of Johnson & Johnson.
The makers of Band-Aid® brand do not manufacture store brand products.
www.band-aid.com.
USA 1-866-JNJ-2873.
©J&J CCI 2011.. Country of origin: Brazil. Warnings: Caution: The packaging of this product contains natural rubber latex, which may cause allergic reactions.."
<human>: Based on the information given in the above text, what are the values for the following attributes title, brand, manufacturer, country of origin and product type. Give the result in a JSON format where the attribute names are the keys and the attribute values are the values for the keys. If the value of an attribute is not present in the information, set the attribute value as 'N/A'.
<assistant>:

""".strip()


tokenizer = AutoTokenizer.from_pretrained("icecreamlabs/Tiger-7B-Instruct", trust_remote_code = True)
model = AutoModelForCausalLM(
    "icecreamlabs/Tiger-7B-Instruct",
    torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
)
inputs = tokenizer(prompt, return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))

"""
Output

{
  "title": "Band-Aid Bandages Comfort-Flex Extra Large All One Size 10 Each",
  "brand": "Band Aid",
  "manufacturer": "Johnson & Johnson",
  "country_of_origin": "Brazil",
  "product_type": "Band-Aid"
}


"""


