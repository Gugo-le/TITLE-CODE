import replicate
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


output = replicate.run(
  "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
  input={"prompt": "students studying in the reading room"}
)

pprint(output)
