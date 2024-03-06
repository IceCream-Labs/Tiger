#####################################################################################################
## Tiger LLM is a finetuned LLM to extracct product data from text in valid json format
## Copyright (C)2023 IceCreamlabs Inc
##
## This program is free software: you can redistribute it and/or modify it under the terms of the 
## GNU Affero General Public License as published by the Free Software Foundation, either version 3
## of the License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
## without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
## See the GNU Affero General Public License for more details.
##
## You should have received a copy of the GNU Affero General Public License along with this program.
## If not, see <https://www.gnu.org/licenses/>.
##
## If your software can interact with users remotely through a computer network, 
## you should also make sure that it provides a way for users to get its source. 
## For example, if your program is a web application, its interface could display a "Source" link 
## that leads users to an archive of the code. There are many ways you could offer source, 
## and different solutions will be better for different programs; 
## see section 13 for the specific requirements.
#####################################################################################################

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


